"""
FractalGenerator v5: Fractal octree generation with VQ-VAE token prediction.

Changes from v4 (all gated behind flags; defaults reproduce v4 behavior):
  - 建议 1: Masked training (`use_masked_training`)
  - 建议 2: Buffer tokens (`buffer_size > 0`)
  - 建议 3: Per-level mid_transformer depth (`mid_blocks_per_level`)
  - 建议 4: Sibling self-attention in expander (`use_sibling_attn`)
  - 建议 5: Toggleable focal loss (`use_focal_loss`)
  - 建议 6: Mask-only split_accuracy metric + per-level breakdown

See FRACTAL_IMPROVEMENT_PLAN.md for design rationale.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm

_octgpt_path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'octgpt'))
if _octgpt_path not in sys.path:
    sys.path.insert(0, _octgpt_path)

import ocnn
from ocnn.octree import Octree
from octgpt.models.octformer import OctFormerStage, OctreeT
from octgpt.models.positional_embedding import SinPosEmb, AbsPosEmb, RMSNorm
from octgpt.utils.utils import depth2batch, batch2depth


def split_recall_precision(pred: torch.Tensor, gt: torch.Tensor):
    """Recall/precision of the split=1 class. Accuracy is dominated by the
    abundant negatives and hides missed splits (false negatives), which are
    exactly what create surface holes: a missed split at depth d kills the
    whole subtree (up to 8^(depth_stop-d) leaves). Empty denominators report
    1.0 so batches without positives don't drag the tracker average down."""
    one = torch.ones((), device=pred.device)
    pos = gt == 1
    pred_pos = pred == 1
    tp = (pred_pos & pos).sum().float()
    npos = pos.sum().float()
    npred = pred_pos.sum().float()
    recall = torch.where(npos > 0, tp / npos.clamp(min=1), one)
    precision = torch.where(npred > 0, tp / npred.clamp(min=1), one)
    return recall, precision


# ============================================================================
# Focal Loss (handles extreme class imbalance in split prediction)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss: -alpha * (1-p)^gamma * log(p)

    Downweights easy negatives (empty nodes), forces model to focus on
    the hard boundary nodes where split decisions actually matter.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        ce = F.cross_entropy(logits, targets, reduction='none')
        return (focal_weight * ce).mean()


# ============================================================================
# Transformer-based Feature Expansion (Local Cross-Attention)
# ============================================================================

class OctantPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        offsets = torch.tensor([
            [-1, -1, -1], [-1, -1, +1], [-1, +1, -1], [-1, +1, +1],
            [+1, -1, -1], [+1, -1, +1], [+1, +1, -1], [+1, +1, +1],
        ], dtype=torch.float32)
        self.register_buffer("offsets", offsets)
        self.proj = nn.Linear(3, dim)

    def forward(self):
        return self.proj(self.offsets)


class LocalCrossAttentionExpander(nn.Module):
    """Parent (N, C) -> children (N*8, C) via local cross-attention.

    Note: with parent context length K=1, the cross-attention softmax over a
    single key degenerates to an MLP. The optional `use_sibling_attn` flag
    adds a real attention among the 8 sibling children to restore the
    inter-child information flow.
    """

    def __init__(self, dim: int, num_heads: int = 4, ffn_ratio: float = 2.0,
                 use_sibling_attn: bool = False):
        super().__init__()
        self.dim = dim
        self.use_sibling_attn = use_sibling_attn
        self.octant_queries = nn.Parameter(torch.zeros(8, dim))
        nn.init.normal_(self.octant_queries, std=0.02)
        self.octant_pos_emb = OctantPositionEmbedding(dim)
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)

        if use_sibling_attn:
            self.norm_sib = LayerNorm(dim)
            self.sibling_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, batch_first=True)

        ffn_dim = int(dim * ffn_ratio)
        self.norm_ffn = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, parent_features):
        N = parent_features.shape[0]
        if N == 0:
            return torch.zeros(0, self.dim, device=parent_features.device)
        if parent_features.dim() == 2:
            context = parent_features.unsqueeze(1)
        else:
            context = parent_features
        q = self.octant_queries.unsqueeze(0).expand(N, -1, -1)
        q = q + self.octant_pos_emb().unsqueeze(0)
        attn_out, _ = self.cross_attn(
            query=self.norm_q(q), key=self.norm_kv(context),
            value=self.norm_kv(context))
        children = q + attn_out

        if self.use_sibling_attn:
            # 8 siblings attend to each other (real attention since K=8)
            sib_normed = self.norm_sib(children)
            sib_out, _ = self.sibling_attn(sib_normed, sib_normed, sib_normed)
            children = children + sib_out

        children = children + self.ffn(self.norm_ffn(children))
        return children.reshape(N * 8, self.dim)


# ============================================================================
# FractalGenerator v5
# ============================================================================

class FractalGenerator(nn.Module):

    def __init__(
        self,
        feature_dim: int = 384,
        num_heads: int = 8,
        blocks_per_level: int = 6,
        full_depth: int = 3,
        depth_stop: int = 6,
        patch_size: int = 2048,
        dilation: int = 2,
        drop_rate: float = 0.1,
        pos_emb_type: str = "SinPosEmb",
        norm_type: str = "LayerNorm",
        use_checkpoint: bool = True,
        use_swin: bool = True,
        split_weight: float = 1.0,
        vq_weight: float = 1.0,
        vq_groups: int = 32,
        vq_size: int = 2,
        expander_num_heads: int = 4,
        # Focal Loss params
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        # Generation threshold for split: scalar, or a list of num_levels
        # values (one per expansion level). A missed split at a coarse level
        # kills the whole subtree (up to 8^(depth_stop-d) leaves -> a hole),
        # while a spurious split can still be rejected at the next level, so
        # coarse levels can afford a lower (recall-biased) threshold.
        split_threshold=0.45,
        # Morphological closing on the generated split (0 = off): at the last
        # expansion level, a node predicted non-split is forced to split when
        # >= split_close_k of its 6 face-adjacent neighbors split. Fills
        # isolated false-negative leaves (small surface holes).
        split_close_k: int = 0,
        # ====== v5 experiment flags ======
        # 建议 1: masked training
        use_masked_training: bool = False,
        mask_ratio_min: float = 0.5,
        # 建议 2: buffer tokens (0 = disabled)
        buffer_size: int = 0,
        # 建议 3: per-level mid_transformer depth (None = all 1, matching v4)
        mid_blocks_per_level: list = None,
        # 建议 4: sibling self-attention in expander
        use_sibling_attn: bool = False,
        # 建议 5: focal loss toggle (False -> standard CE)
        use_focal_loss: bool = True,
        # 建议 6: mask-only metric reporting (default ON; set metric_mask_ratio
        #         for the masking applied when use_masked_training is OFF, so
        #         baselines can still report a mask-only number comparable to
        #         OctGPT's metric)
        metric_mask_ratio: float = 0.7,
        # class-conditional generation: inject a per-class embedding at the root
        # so the model can distinguish shapes instead of learning the marginal
        # split distribution. Zero-initialised so a warm-started baseline ckpt
        # starts out exactly equivalent to the unconditional model.
        use_class_cond: bool = False,
        num_classes: int = 5,
        # re-inject the class embedding at every expansion level (not just root)
        # so the conditioning signal does not dilute by the time it reaches the
        # fine levels, where split accuracy is lowest.
        cond_every_level: bool = False,
        # at generation, sample split ~ Bernoulli(p) instead of thresholding
        # p > split_threshold. Adds diversity and avoids a whole branch dying
        # when p sits just below the threshold (the "chair collapse").
        split_sample: bool = False,
        # VAE-style latent: encode the GT shape (pooled frozen VQ-VAE code) into
        # z, inject at the root. Captures intra-class variation and raises the
        # information ceiling. At generation z ~ N(0, I). z_proj is zero-init so
        # a warm-started baseline starts equivalent (latent ignored initially).
        use_latent: bool = False,
        latent_dim: int = 256,
        vq_code_dim: int = 32,
        kl_weight: float = 1e-4,
        # spatially-aware latent: pool the GT VQ code per full_depth cell
        # (8^full_depth cells) instead of one global mean. A global mean is
        # structure-blind (average surface descriptor), which is why the Q4
        # latent never influenced the split; per-cell pooling puts coarse
        # spatial layout into z, exactly the information the depth-3 split
        # head is missing.
        latent_spatial_pool: bool = False,
        # per-level split loss weights (None = uniform, v4 behavior). A missed
        # split at depth d kills 8^(depth_stop-d) leaves, so coarse levels
        # deserve more weight, e.g. [4, 2, 1]. Diagnosed depth-3 FN 21% is the
        # structural bottleneck in multi-class generation.
        level_loss_weights: list = None,
        # leaf VQ MaskGIT: train the leaf VQ head to predict masked tokens from
        # visible-token context (vq_proj embeds known tokens), so at generation
        # the surface tokens can be refined over a few iterative reveal steps
        # (OctGPT-style, but only at the leaf and only leaf_vq_iters steps).
        leaf_vq_mask: bool = False,
        leaf_vq_iters: int = 1,
        leaf_vq_temp: float = 1.0,
        # split-structure MaskGIT: at each level, embed revealed splits as input
        # (split_emb) and predict masked ones; at generation, refine the split
        # structure over split_iters iterative reveal steps. Attacks split
        # quality (missing depth-6 nodes -> surface holes) directly. Few steps
        # (4-8) -> still ~10-40x faster than OctGPT's 576.
        split_mask: bool = False,
        split_iters: int = 1,
        split_temp: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.full_depth = full_depth
        self.depth_stop = depth_stop
        self.num_levels = depth_stop - full_depth
        self.patch_size = patch_size
        self.dilation = dilation
        self.use_swin = use_swin
        self.split_weight = split_weight
        self.vq_weight = vq_weight
        self.vq_groups = vq_groups
        self.vq_size = vq_size
        if isinstance(split_threshold, (list, tuple)):
            assert len(split_threshold) == self.num_levels, (
                f"split_threshold list length ({len(split_threshold)}) "
                f"must equal num_levels ({self.num_levels})")
            split_threshold = [float(t) for t in split_threshold]
        self.split_threshold = split_threshold
        self.split_close_k = split_close_k

        # v5 flags
        self.use_masked_training = use_masked_training
        self.mask_ratio_min = mask_ratio_min
        self.buffer_size = buffer_size
        self.use_sibling_attn = use_sibling_attn
        self.use_focal_loss = use_focal_loss
        self.metric_mask_ratio = metric_mask_ratio

        # Per-level split loss weights
        if level_loss_weights is not None:
            assert len(level_loss_weights) == self.num_levels, (
                f"level_loss_weights length ({len(level_loss_weights)}) "
                f"must equal num_levels ({self.num_levels})")
            level_loss_weights = [float(w) for w in level_loss_weights]
        self.level_loss_weights = level_loss_weights

        # Per-level mid_transformer depths
        if mid_blocks_per_level is None:
            mid_blocks_per_level = [1] * self.num_levels
        else:
            assert len(mid_blocks_per_level) == self.num_levels, (
                f"mid_blocks_per_level length ({len(mid_blocks_per_level)}) "
                f"must equal num_levels ({self.num_levels})")
        self.mid_blocks_per_level = mid_blocks_per_level

        PosEmb = eval(pos_emb_type)
        Norm = eval(norm_type)

        # ---- Focal Loss for split (kept for back-compat; gated by flag) ----
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # ---- learnable root features ----
        self.root_embedding = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.root_embedding, std=0.02)

        # ---- class-conditional root injection ----
        self.use_class_cond = use_class_cond
        self.num_classes = num_classes
        self.cond_every_level = cond_every_level
        self.split_sample = split_sample
        if self.use_class_cond:
            self.class_embedding = nn.Embedding(num_classes, feature_dim)
            # zeroed after self.apply(self._init_weights) below

        # ---- VAE latent: encode pooled GT VQ code -> z, inject at root ----
        self.use_latent = use_latent
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.latent_spatial_pool = latent_spatial_pool
        self.latent_pool_cells = (
            8 ** full_depth if latent_spatial_pool else 1)
        if self.use_latent:
            z_in_dim = vq_code_dim * self.latent_pool_cells
            self.z_mu = nn.Linear(z_in_dim, latent_dim)
            self.z_logvar = nn.Linear(z_in_dim, latent_dim)
            # z_proj zeroed after init so warm-start ignores z initially
            self.z_proj = nn.Linear(latent_dim, feature_dim)

        # ---- leaf VQ MaskGIT ----
        self.leaf_vq_mask = leaf_vq_mask
        self.leaf_vq_iters = leaf_vq_iters
        self.leaf_vq_temp = leaf_vq_temp
        if self.leaf_vq_mask:
            self.leaf_vq_mask_emb = nn.Parameter(torch.zeros(1, feature_dim))

        # ---- split-structure MaskGIT ----
        self.split_mask = split_mask
        self.split_iters = split_iters
        self.split_temp = split_temp
        if self.split_mask:
            self.split_emb = nn.Embedding(2, feature_dim)
            self.split_mask_emb = nn.Parameter(torch.zeros(1, feature_dim))

        # ---- spatial position projection ----
        self.pos_proj = nn.Linear(3, feature_dim)

        # ---- masked-training: learnable mask token ----
        if self.use_masked_training:
            self.mask_emb = nn.Parameter(torch.zeros(1, feature_dim))
            nn.init.normal_(self.mask_emb, std=0.02)

        # ---- buffer tokens (always-visible global context channel) ----
        if self.buffer_size > 0:
            self.buffer_emb = nn.Parameter(
                torch.zeros(self.buffer_size, feature_dim))
            nn.init.normal_(self.buffer_emb, std=0.02)

        # ---- per-level mid transformers ----
        self.mid_transformers = nn.ModuleList([
            OctFormerStage(
                dim=feature_dim, num_heads=num_heads,
                num_blocks=self.mid_blocks_per_level[i],
                patch_size=patch_size, dilation=dilation,
                attn_drop=drop_rate, proj_drop=drop_rate, dropout=drop_rate,
                nempty=False, use_checkpoint=use_checkpoint,
                use_swin=use_swin, pos_emb=PosEmb, norm_layer=Norm)
            for i in range(self.num_levels)
        ])
        self.mid_norms = nn.ModuleList([
            Norm(feature_dim) for _ in range(self.num_levels)
        ])

        # ---- leaf-level transformer ----
        self.leaf_transformer = OctFormerStage(
            dim=feature_dim, num_heads=num_heads,
            num_blocks=blocks_per_level, patch_size=patch_size,
            dilation=dilation, attn_drop=drop_rate,
            proj_drop=drop_rate, dropout=drop_rate,
            nempty=False, use_checkpoint=use_checkpoint,
            use_swin=use_swin, pos_emb=PosEmb, norm_layer=Norm)
        self.leaf_norm = Norm(feature_dim)

        # ---- per-level split heads ----
        self.split_heads = nn.ModuleList([
            nn.Linear(feature_dim, 2) for _ in range(self.num_levels)
        ])

        # ---- per-level feature expanders ----
        self.feature_expanders = nn.ModuleList([
            LocalCrossAttentionExpander(
                dim=feature_dim, num_heads=expander_num_heads,
                use_sibling_attn=use_sibling_attn)
            for _ in range(self.num_levels)
        ])

        # ---- shared child-position embedding ----
        self.child_pos_emb = nn.Parameter(torch.zeros(8, feature_dim))
        nn.init.normal_(self.child_pos_emb, std=0.02)

        # ---- VQ token prediction head ----
        self.vq_head = nn.Linear(feature_dim, vq_groups * vq_size)
        self.vq_proj = nn.Linear(vq_groups, feature_dim)

        self.apply(self._init_weights)

        # zero-init class embedding (adaLN-zero style): a warm-started baseline
        # ckpt then starts byte-for-byte equivalent to the unconditional model,
        # and the conditioning signal is learned from there.
        if self.use_class_cond:
            nn.init.zeros_(self.class_embedding.weight)
        # zero-init z_proj so latent injection starts as a no-op (warm-start
        # equivalence); the model learns to use z as KL anneals it in.
        if self.use_latent:
            nn.init.zeros_(self.z_proj.weight)
            nn.init.zeros_(self.z_proj.bias)
        # zero token-embedding path so leaf starts equivalent to one-shot vq_head
        # (visible tokens contribute nothing at init; learned via attention grad)
        if self.leaf_vq_mask:
            nn.init.zeros_(self.leaf_vq_mask_emb)
            nn.init.zeros_(self.vq_proj.weight)
            nn.init.zeros_(self.vq_proj.bias)
        # zero split-token path so warm-start starts equivalent to single-pass
        if self.split_mask:
            nn.init.zeros_(self.split_emb.weight)
            nn.init.zeros_(self.split_mask_emb)
        # zero sibling-attn output so warm-start from a no-sibling ckpt starts
        # equivalent (children += sib_out, sib_out=0 at init); learned from there
        if self.use_sibling_attn:
            for exp in self.feature_expanders:
                nn.init.zeros_(exp.sibling_attn.out_proj.weight)
                nn.init.zeros_(exp.sibling_attn.out_proj.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pos_embed(self, octree, depth):
        ox, oy, oz, ob = octree.xyzb(depth)
        scale = 2 ** depth
        pos = torch.stack([
            ox.float() / scale, oy.float() / scale, oz.float() / scale,
        ], dim=-1)
        return self.pos_proj(pos)

    def _run_mid_transformer(self, features, octree, depth, lvl):
        """Run mid_transformer with optional buffer-token prefix."""
        B = octree.batch_size

        if self.buffer_size > 0:
            # Prepend B copies of the buffer
            buffer = self.buffer_emb.unsqueeze(0).expand(B, -1, -1).reshape(
                B * self.buffer_size, self.feature_dim)
            feat_with_buf = torch.cat([buffer, features], dim=0)
            data_length = feat_with_buf.shape[0]
        else:
            feat_with_buf = features
            data_length = features.shape[0]

        octreeT = OctreeT(
            octree, data_length, self.patch_size, self.dilation,
            nempty=False, depth_list=[depth],
            buffer_size=self.buffer_size,
            use_swin=self.use_swin)

        feat = depth2batch(feat_with_buf, octreeT.indices)
        feat = self.mid_transformers[lvl](feat, octreeT, context=None)
        feat = batch2depth(feat, octreeT.indices)

        # Strip the buffer prefix from the output
        if self.buffer_size > 0:
            feat = feat[B * self.buffer_size:]

        return self.mid_norms[lvl](feat)

    def _run_leaf_transformer(self, features, octree, depth):
        octreeT = OctreeT(
            octree, features.shape[0], self.patch_size, self.dilation,
            nempty=False, depth_list=[depth], buffer_size=0,
            use_swin=self.use_swin)
        feat = depth2batch(features, octreeT.indices)
        feat = self.leaf_transformer(feat, octreeT, context=None)
        feat = batch2depth(feat, octreeT.indices)
        return self.leaf_norm(feat)

    def _leaf_vq_train(self, features, octree, gt_indices):
        """MaskGIT leaf training: hide a random subset of nodes' VQ tokens,
        feed visible tokens (via vq_proj) as context, predict the masked ones.
        Returns (vq_logits [N,G,size], vq_mask [N] bool over hidden nodes)."""
        n = features.shape[0]
        ratio = float(torch.empty(1).uniform_(0.1, 1.0).item())
        k = max(1, int(n * ratio))
        perm = torch.randperm(n, device=features.device)
        vq_mask = torch.zeros(n, dtype=torch.bool, device=features.device)
        vq_mask[perm[:k]] = True
        tok = self.vq_proj(gt_indices.float())
        leaf_in = features + torch.where(
            vq_mask.unsqueeze(1), self.leaf_vq_mask_emb.expand(n, -1), tok)
        feat = self._run_leaf_transformer(leaf_in, octree, self.depth_stop)
        logits = self.vq_head(feat).reshape(n, self.vq_groups, self.vq_size)
        return logits, vq_mask

    @torch.no_grad()
    def _leaf_vq_generate(self, features, octree, temperature):
        """Iterative MaskGIT decode of leaf VQ tokens (leaf_vq_iters steps,
        cosine reveal schedule, confidence-based). Returns indices [N,G]."""
        import math
        n = features.shape[0]
        device = features.device
        iters = max(1, self.leaf_vq_iters)
        t = temperature if temperature > 0 else self.leaf_vq_temp
        indices = torch.zeros(n, self.vq_groups, dtype=torch.long, device=device)
        known = torch.zeros(n, dtype=torch.bool, device=device)
        samp = indices
        for i in range(iters):
            tok = self.vq_proj(indices.float())
            leaf_in = features + torch.where(
                known.unsqueeze(1), tok, self.leaf_vq_mask_emb.expand(n, -1))
            feat = self._run_leaf_transformer(leaf_in, octree, self.depth_stop)
            logits = self.vq_head(feat).reshape(n, self.vq_groups, self.vq_size)
            probs = F.softmax(logits / max(t, 1e-6), dim=-1)
            samp = torch.multinomial(
                probs.reshape(-1, self.vq_size), 1).reshape(n, self.vq_groups)
            conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1).mean(dim=1)
            target = n if i == iters - 1 else int(
                round(n * (1.0 - math.cos(math.pi / 2 * (i + 1) / iters))))
            newly = target - int(known.sum().item())
            if newly > 0:
                conf_avail = conf.masked_fill(known, -1.0)
                avail = int((~known).sum().item())
                topk = torch.topk(conf_avail, min(newly, avail)).indices
                indices[topk] = samp[topk]
                known[topk] = True
        if not bool(known.all()):
            indices[~known] = samp[~known]
        return indices

    @torch.no_grad()
    def _split_generate(self, features, octree, depth, lvl):
        """Iterative MaskGIT decode of the split at one level (split_iters steps,
        cosine reveal, confidence-based). Returns (split [n] long, feat [n,dim]
        from a final fully-revealed pass for expansion)."""
        import math
        n = features.shape[0]
        device = features.device
        iters = max(1, self.split_iters)
        t = self.split_temp if self.split_temp > 0 else 1.0
        split = torch.zeros(n, dtype=torch.long, device=device)
        known = torch.zeros(n, dtype=torch.bool, device=device)
        for k in range(iters):
            tok = torch.where(known.unsqueeze(1), self.split_emb(split),
                              self.split_mask_emb.expand(n, -1))
            feat = self._run_mid_transformer(features + tok, octree, depth, lvl)
            probs = F.softmax(self.split_heads[lvl](feat) / t, dim=-1)
            samp = torch.bernoulli(probs[:, 1]).long()
            conf = probs.gather(1, samp.unsqueeze(1)).squeeze(1)
            target = n if k == iters - 1 else int(
                round(n * (1.0 - math.cos(math.pi / 2 * (k + 1) / iters))))
            newly = target - int(known.sum().item())
            if newly > 0:
                conf_avail = conf.masked_fill(known, -1.0)
                topk = torch.topk(conf_avail, min(newly, int((~known).sum().item()))).indices
                split[topk] = samp[topk]
                known[topk] = True
        if not bool(known.all()):
            split[~known] = torch.bernoulli(probs[:, 1][~known]).long()
        # final fully-revealed pass -> expansion features (matches training)
        feat = self._run_mid_transformer(
            features + self.split_emb(split), octree, depth, lvl)
        return split, feat

    def _threshold_for(self, lvl: int) -> float:
        if isinstance(self.split_threshold, (list, tuple)):
            return self.split_threshold[lvl]
        return self.split_threshold

    @torch.no_grad()
    def _close_split_holes(self, split, octree, depth):
        """Force-split nodes with >= split_close_k face-adjacent split
        neighbors. Only fills isolated misses at this depth; a hole whose
        parent was already dropped at a coarser level has no node here and
        cannot be recovered."""
        k = int(self.split_close_k)
        if k <= 0 or split.numel() == 0:
            return split
        split_mask = split == 1
        if not split_mask.any() or split_mask.all():
            return split
        x, y, z, b = octree.xyzb(depth)
        pos = torch.stack([x, y, z], dim=1).long()
        S = 2 ** depth
        base = b.long() * (S ** 3)
        keys = base + (pos[:, 0] * S + pos[:, 1]) * S + pos[:, 2]
        split_keys = keys[split_mask]
        offsets = torch.tensor(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0],
             [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=torch.long, device=split.device)
        nb = pos.unsqueeze(1) + offsets.unsqueeze(0)          # [n, 6, 3]
        valid = ((nb >= 0) & (nb < S)).all(dim=-1)            # [n, 6]
        nb_keys = base.view(-1, 1) + (
            nb[..., 0] * S + nb[..., 1]) * S + nb[..., 2]
        hit = torch.isin(nb_keys, split_keys) & valid
        force = (~split_mask) & (hit.sum(dim=1) >= k)
        if force.any():
            split = split.clone()
            split[force] = 1
        return split

    def _expand_features(self, features, split_mask, level_idx):
        parents = features[split_mask]
        n = parents.shape[0]
        if n == 0:
            return torch.zeros(0, self.feature_dim, device=features.device)
        children = self.feature_expanders[level_idx](parents)
        children = children.view(n, 8, self.feature_dim)
        children = children + self.child_pos_emb.unsqueeze(0)
        return children.reshape(n * 8, self.feature_dim)

    def _sample_mask(self, n: int, device, force_full: bool = False
                     ) -> torch.Tensor:
        """Sample a bool mask of length n. When `force_full`, mask=True
        everywhere (used at inference to keep the path identical to the
        non-masked code path)."""
        if force_full:
            return torch.ones(n, dtype=torch.bool, device=device)
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        if self.use_masked_training:
            # Truncated normal in [mask_ratio_min, 1.0], mean=1.0, std=0.25.
            # Uses torch RNG so DDP rank seeding and manual_seed apply.
            raw = torch.empty(1).normal_(mean=1.0, std=0.25)
            mask_ratio = float(raw.clamp(self.mask_ratio_min, 1.0).item())
        else:
            mask_ratio = self.metric_mask_ratio
        num_masked = max(1, int(n * mask_ratio))
        orders = torch.randperm(n, device=device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[orders[:num_masked]] = True
        return mask

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def _inject_cond(self, features, octree, depth, label, z=None):
        """Add per-class embedding and/or latent z to each node, gathered by
        batch id. No-op for whichever conditioning is off, so the unconditional
        path is byte-for-byte unchanged."""
        bid = None
        if self.use_class_cond and label is not None:
            bid = octree.batch_id(depth, nempty=False).long()
            features = features + self.class_embedding(label[bid])
        if self.use_latent and z is not None:
            if bid is None:
                bid = octree.batch_id(depth, nempty=False).long()
            features = features + self.z_proj(z)[bid]
        return features

    def _encode_latent(self, vq_code, octree, batch_size):
        """Encode pooled GT VQ code into a latent z (VAE posterior) + KL.

        Mean-pools the frozen VQ-VAE leaf code per sample (globally, or per
        full_depth cell when latent_spatial_pool so z keeps coarse layout),
        then maps to (mu, logvar) and reparameterises. Returns (z, kl_loss)."""
        bid = octree.batch_id(self.depth_stop, nempty=False).long()
        dim = vq_code.shape[1]
        cells = self.latent_pool_cells
        if self.latent_spatial_pool:
            x, y, z_, _ = octree.xyzb(self.depth_stop)
            shift = self.depth_stop - self.full_depth
            S = 2 ** self.full_depth
            cell = ((x.long() >> shift) * S + (y.long() >> shift)) * S + \
                (z_.long() >> shift)
            slot = bid * cells + cell
        else:
            slot = bid
        pooled = torch.zeros(batch_size * cells, dim, device=vq_code.device,
                             dtype=vq_code.dtype)
        pooled.index_add_(0, slot, vq_code)
        counts = torch.zeros(batch_size * cells, device=vq_code.device,
                             dtype=vq_code.dtype)
        counts.index_add_(0, slot, torch.ones_like(slot, dtype=vq_code.dtype))
        pooled = pooled / counts.clamp(min=1).unsqueeze(1)
        pooled = pooled.reshape(batch_size, cells * dim)
        mu = self.z_mu(pooled)
        logvar = self.z_logvar(pooled)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl

    def forward(self, octree_gt, vqvae=None, label=None):
        device = octree_gt.device
        output = {}

        # ---- 1. Extract GT VQ targets ----
        if vqvae is not None:
            with torch.no_grad():
                vq_code = vqvae.extract_code(octree_gt)
                _, gt_indices, _ = vqvae.quantizer(vq_code)
        else:
            gt_indices = None

        # ---- 1b. Encode latent z from GT (VAE posterior) ----
        z = None
        kl_loss = torch.tensor(0.0, device=device)
        if self.use_latent and vqvae is not None:
            z, kl_loss = self._encode_latent(
                vq_code, octree_gt, octree_gt.batch_size)

        # ---- 2. Init features with position encoding ----
        features = self.root_embedding.expand(
            octree_gt.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree_gt, self.full_depth)
        features = self._inject_cond(
            features, octree_gt, self.full_depth, label, z)

        total_split_loss = torch.tensor(0.0, device=device)
        per_level_acc_mask = []  # mask-only accuracy (new primary metric)
        per_level_acc_all = []   # all-position accuracy (legacy v4 metric)
        per_level_rec_mask = []   # split=1 recall, mask-only
        per_level_prec_mask = []  # split=1 precision, mask-only
        per_level_rec_all = []    # split=1 recall, all positions
        per_level_prec_all = []   # split=1 precision, all positions

        # ---- 3. Fractal expansion ----
        for lvl in range(self.num_levels):
            d = self.full_depth + lvl
            nnum_d = octree_gt.nnum[d]
            n = features.shape[0]

            assert n == nnum_d, \
                f"Shape mismatch at depth {d}: features={n}, nnum={nnum_d}"

            # Sample a mask (always, for metric purposes; used for loss only
            # when use_masked_training is on)
            mask = self._sample_mask(n, device, force_full=not self.training)

            apply_mask = (self.use_masked_training and self.training and n > 0)
            split_mask_active = self.split_mask and self.training and n > 0

            if split_mask_active:
                # Split MaskGIT: embed revealed GT splits as input, predict the
                # masked ones. Two passes: a fully-revealed pass feeds expansion
                # (matches generation's final state), a masked pass gives the
                # loss. split_emb/split_mask_emb are zero-init so warm-start
                # starts equivalent to single-pass.
                # HEAVY mask ratio (0.8-1.0). Near-fully-masked removes the
                # "copy revealed neighbours" shortcut, forcing prediction from
                # features+conditioning (the hard, useful regime that the early
                # generation iterations hit). Light masking (earlier 0.05-1.0)
                # let the model cheat off revealed GT and degraded single-pass.
                ratio = float(torch.empty(1).uniform_(0.8, 1.0).item())
                k_m = max(1, int(round(n * ratio)))
                perm = torch.randperm(n, device=device)
                mask = torch.zeros(n, dtype=torch.bool, device=device)
                mask[perm[:k_m]] = True
                gt_split_d = (octree_gt.children[d] >= 0).long()
                feat_rev = self._run_mid_transformer(
                    features + self.split_emb(gt_split_d), octree_gt, d, lvl)
                inp = features + torch.where(
                    mask.unsqueeze(1), self.split_mask_emb.expand(n, -1),
                    self.split_emb(gt_split_d))
                features_for_pred = self._run_mid_transformer(
                    inp, octree_gt, d, lvl)
                features = feat_rev
            elif apply_mask:
                # Masked path: mask BEFORE mid_transformer so the transformer
                # learns to infer masked nodes from context (BERT-style).
                # Keep a clean copy for expansion (matches generate() flow).
                features_clean = features
                features_masked = torch.where(
                    mask.unsqueeze(1),
                    self.mask_emb.expand(n, -1),
                    features,
                )
                if n > 0:
                    features_masked = self._run_mid_transformer(
                        features_masked, octree_gt, d, lvl)
                    features_clean = self._run_mid_transformer(
                        features_clean, octree_gt, d, lvl)
                features_for_pred = features_masked
                features = features_clean
            else:
                # Unmasked path (v4 behavior): single mid_transformer pass
                if n > 0:
                    features = self._run_mid_transformer(
                        features, octree_gt, d, lvl)
                features_for_pred = features

            # Predict split
            logits = self.split_heads[lvl](features_for_pred)
            gt_split = (octree_gt.children[d] >= 0).long()

            # Loss (respects use_focal_loss in all branches)
            if (apply_mask or split_mask_active) and mask.any():
                if self.use_focal_loss:
                    loss_lvl = self.focal_loss(logits[mask], gt_split[mask])
                else:
                    loss_lvl = F.cross_entropy(logits[mask], gt_split[mask])
            elif self.use_focal_loss:
                loss_lvl = self.focal_loss(logits, gt_split)
            else:
                loss_lvl = F.cross_entropy(logits, gt_split)
            if self.level_loss_weights is not None:
                loss_lvl = loss_lvl * self.level_loss_weights[lvl]
            total_split_loss = total_split_loss + loss_lvl

            # Metrics: report both new (mask-only) and legacy (all-pos)
            with torch.no_grad():
                pred_split = logits.argmax(-1)
                if n > 0:
                    acc_all_lvl = (pred_split == gt_split).float().mean()
                else:
                    acc_all_lvl = torch.tensor(0.0, device=device)
                if mask.any():
                    acc_mask_lvl = (
                        pred_split[mask] == gt_split[mask]).float().mean()
                    rec_lvl, prec_lvl = split_recall_precision(
                        pred_split[mask], gt_split[mask])
                else:
                    acc_mask_lvl = acc_all_lvl
                    rec_lvl, prec_lvl = split_recall_precision(
                        pred_split, gt_split)
                rec_all_lvl, prec_all_lvl = split_recall_precision(
                    pred_split, gt_split)
                per_level_acc_all.append(acc_all_lvl)
                per_level_acc_mask.append(acc_mask_lvl)
                per_level_rec_mask.append(rec_lvl)
                per_level_prec_mask.append(prec_lvl)
                per_level_rec_all.append(rec_all_lvl)
                per_level_prec_all.append(prec_all_lvl)
                output[f'split_acc_lvl{lvl}'] = acc_mask_lvl
                output[f'split_recall_lvl{lvl}'] = rec_lvl
                output[f'split_prec_lvl{lvl}'] = prec_lvl

            # Expand using teacher-forced split
            split_mask_gt = (gt_split == 1)
            child_features = self._expand_features(features, split_mask_gt, lvl)
            child_features = child_features + self._get_pos_embed(
                octree_gt, d + 1)
            if self.cond_every_level:
                child_features = self._inject_cond(
                    child_features, octree_gt, d + 1, label, z)
            features = child_features

        # ---- 4. Leaf transformer + VQ token prediction ----
        nnum_leaf = octree_gt.nnum[self.depth_stop]
        assert features.shape[0] == nnum_leaf, "Final leaf count mismatch!"

        has_leaf = features.shape[0] > 0
        vq_mask = None
        if has_leaf and gt_indices is not None and \
                self.leaf_vq_mask and self.training:
            vq_logits, vq_mask = self._leaf_vq_train(
                features, octree_gt, gt_indices)
        elif has_leaf:
            leaf_feat = self._run_leaf_transformer(
                features, octree_gt, self.depth_stop)
            vq_logits = self.vq_head(leaf_feat).reshape(
                -1, self.vq_groups, self.vq_size)
        else:
            vq_logits = None

        loss_norm = (sum(self.level_loss_weights)
                     if self.level_loss_weights is not None
                     else self.num_levels)
        output["split_loss"] = total_split_loss / max(loss_norm, 1)
        output["split_accuracy"] = (
            torch.stack(per_level_acc_mask).mean()
            if per_level_acc_mask else torch.tensor(0.0, device=device)
        )
        output["split_accuracy_all"] = (
            torch.stack(per_level_acc_all).mean()
            if per_level_acc_all else torch.tensor(0.0, device=device)
        )
        if per_level_rec_mask:
            output["split_recall"] = torch.stack(per_level_rec_mask).mean()
            output["split_precision"] = torch.stack(per_level_prec_mask).mean()
            output["split_recall_all"] = torch.stack(per_level_rec_all).mean()
            output["split_precision_all"] = (
                torch.stack(per_level_prec_all).mean())

        # ---- 5. VQ loss (mask-only when leaf MaskGIT is on) ----
        if vq_logits is not None and gt_indices is not None:
            if vq_mask is not None and vq_mask.any():
                logit_sel = vq_logits[vq_mask].reshape(-1, self.vq_size)
                gt_sel = gt_indices[vq_mask].reshape(-1).long()
            else:
                logit_sel = vq_logits.reshape(-1, self.vq_size)
                gt_sel = gt_indices.reshape(-1).long()
            output["vq_loss"] = F.cross_entropy(logit_sel, gt_sel)
            with torch.no_grad():
                output["vq_accuracy"] = (
                    logit_sel.argmax(-1) == gt_sel).float().mean()
        else:
            output["vq_loss"] = torch.tensor(0.0, device=device)
            output["vq_accuracy"] = torch.tensor(0.0, device=device)

        output["loss"] = (self.split_weight * output["split_loss"]
                          + self.vq_weight * output["vq_loss"])
        if self.use_latent:
            output["kl_loss"] = kl_loss
            output["loss"] = output["loss"] + self.kl_weight * kl_loss
        return output

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def teacher_forced_probs(self, octree_gt, vqvae=None, label=None,
                             posterior_z=True):
        """Inference-path pass on the GT octree, expanded with the GT split.

        Unlike forward(), this uses the generation-time feature path (no
        masking) so the returned per-level P(split) can be judged under the
        actual threshold rule: false negatives here are exactly the nodes a
        free generation run would drop, isolated from cascade effects.
        Returns a list of (probs_split1 [n], gt_split [n]) per level."""
        device = octree_gt.device
        z = None
        if self.use_latent:
            if posterior_z and vqvae is not None:
                vq_code = vqvae.extract_code(octree_gt)
                z, _ = self._encode_latent(
                    vq_code, octree_gt, octree_gt.batch_size)
            else:
                z = torch.randn(
                    octree_gt.batch_size, self.latent_dim, device=device)

        features = self.root_embedding.expand(
            octree_gt.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree_gt, self.full_depth)
        features = self._inject_cond(
            features, octree_gt, self.full_depth, label, z)

        out = []
        for lvl in range(self.num_levels):
            d = self.full_depth + lvl
            n = features.shape[0]
            gt_split = (octree_gt.children[d] >= 0).long()
            if n == 0:
                out.append((torch.zeros(0, device=device), gt_split))
                break
            if self.split_mask:
                # first generation iteration sees an all-masked split input
                feat = self._run_mid_transformer(
                    features + self.split_mask_emb.expand(n, -1),
                    octree_gt, d, lvl)
            else:
                feat = self._run_mid_transformer(features, octree_gt, d, lvl)
            probs = F.softmax(self.split_heads[lvl](feat), dim=-1)[:, 1]
            out.append((probs, gt_split))

            if self.split_mask:
                # expansion features come from the fully-revealed pass,
                # matching both training and generation's final state
                feat = self._run_mid_transformer(
                    features + self.split_emb(gt_split), octree_gt, d, lvl)
            features = self._expand_features(feat, gt_split == 1, lvl)
            features = features + self._get_pos_embed(octree_gt, d + 1)
            if self.cond_every_level:
                features = self._inject_cond(
                    features, octree_gt, d + 1, label, z)
        return out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch_size=1, device="cuda", temperature=0.8,
                 vqvae=None, label=None, z=None):
        """Generate with threshold-based split and temperature VQ sampling.

        Inference path: NO masking applied (mask_emb never touches features).
        The mid_transformer receives raw features; this is equivalent to the
        v4 inference path even when `use_masked_training=True`."""
        octree = ocnn.octree.init_octree(
            self.depth_stop, self.full_depth, batch_size, device)

        # latent prior sample (no GT at generation); an explicit z overrides,
        # e.g. a GT-posterior z for reconstruction-style diagnostics
        if self.use_latent:
            if z is None:
                z = torch.randn(batch_size, self.latent_dim, device=device)
        else:
            z = None

        features = self.root_embedding.expand(
            octree.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree, self.full_depth)
        features = self._inject_cond(
            features, octree, self.full_depth, label, z)

        for lvl in range(self.num_levels):
            d = self.full_depth + lvl

            if features.shape[0] > 0 and self.split_mask:
                # iterative MaskGIT split refinement; feat is the revealed-pass
                # features used for expansion
                split, feat = self._split_generate(features, octree, d, lvl)
            else:
                if features.shape[0] > 0:
                    features = self._run_mid_transformer(
                        features, octree, d, lvl)
                logits = self.split_heads[lvl](features)
                probs = F.softmax(logits, dim=-1)
                if self.split_sample:
                    split = torch.bernoulli(probs[:, 1]).long()
                else:
                    split = (probs[:, 1] > self._threshold_for(lvl)).long()
                feat = features

            if lvl == self.num_levels - 1:
                split = self._close_split_holes(split, octree, d)

            octree.octree_split(split, d)
            octree.octree_grow(d + 1)

            features = self._expand_features(feat, split.bool(), lvl)
            if features.shape[0] == 0:
                break

            features = features + self._get_pos_embed(octree, d + 1)
            if self.cond_every_level:
                features = self._inject_cond(features, octree, d + 1, label, z)

        if features.shape[0] > 0 and self.leaf_vq_mask:
            indices = self._leaf_vq_generate(features, octree, temperature)
        elif features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree, self.depth_stop)
            vq_logits = self.vq_head(features)
            vq_logits = vq_logits.reshape(-1, self.vq_groups, self.vq_size)
            if temperature > 0:
                probs = F.softmax(vq_logits / temperature, dim=-1)
                indices = torch.multinomial(
                    probs.reshape(-1, self.vq_size), 1).reshape(
                        -1, self.vq_groups)
            else:
                indices = vq_logits.argmax(-1)
        else:
            indices = torch.zeros(
                0, self.vq_groups, dtype=torch.long, device=device)

        if vqvae is not None:
            vq_code = vqvae.quantizer.extract_code(indices)
        else:
            vq_code = indices.float()

        return octree, vq_code
