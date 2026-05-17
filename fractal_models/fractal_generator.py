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
import scipy.stats as stats

from torch.nn import LayerNorm

_octgpt_path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'octgpt'))
if _octgpt_path not in sys.path:
    sys.path.insert(0, _octgpt_path)

import ocnn
from ocnn.octree import Octree
from octgpt.models.octformer import OctFormerStage, OctreeT
from octgpt.models.positional_embedding import SinPosEmb, AbsPosEmb, RMSNorm
from octgpt.utils.utils import depth2batch, batch2depth


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
        # Generation threshold for split
        split_threshold: float = 0.45,
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
        self.split_threshold = split_threshold

        # v5 flags
        self.use_masked_training = use_masked_training
        self.mask_ratio_min = mask_ratio_min
        self.buffer_size = buffer_size
        self.use_sibling_attn = use_sibling_attn
        self.use_focal_loss = use_focal_loss
        self.metric_mask_ratio = metric_mask_ratio

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

        # ---- spatial position projection ----
        self.pos_proj = nn.Linear(3, feature_dim)

        # ---- masked-training: learnable mask token + ratio sampler ----
        if self.use_masked_training:
            self.mask_emb = nn.Parameter(torch.zeros(1, feature_dim))
            nn.init.normal_(self.mask_emb, std=0.02)
            self.mask_ratio_generator = stats.truncnorm(
                (self.mask_ratio_min - 1.0) / 0.25, 0.0,
                loc=1.0, scale=0.25)

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
            mask_ratio = float(self.mask_ratio_generator.rvs(1)[0])
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

    def forward(self, octree_gt, vqvae=None):
        device = octree_gt.device
        output = {}

        # ---- 1. Extract GT VQ targets ----
        if vqvae is not None:
            with torch.no_grad():
                vq_code = vqvae.extract_code(octree_gt)
                _, gt_indices, _ = vqvae.quantizer(vq_code)
        else:
            gt_indices = None

        # ---- 2. Init features with position encoding ----
        features = self.root_embedding.expand(
            octree_gt.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree_gt, self.full_depth)

        total_split_loss = torch.tensor(0.0, device=device)
        per_level_acc_mask = []  # mask-only accuracy (new primary metric)
        per_level_acc_all = []   # all-position accuracy (legacy v4 metric)

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

            # Apply mask to features only when training with masking
            apply_mask = (self.use_masked_training and self.training and n > 0)
            if apply_mask:
                features_for_pred = torch.where(
                    mask.unsqueeze(1),
                    self.mask_emb.expand(n, -1),
                    features,
                )
            else:
                features_for_pred = features

            # Node communication (mid_transformer)
            if n > 0:
                features_for_pred = self._run_mid_transformer(
                    features_for_pred, octree_gt, d, lvl)

            # When NOT masking, propagate mid_transformer output to `features`
            # so expansion benefits (v4 behavior). When masking, keep raw
            # features for expansion to avoid propagating mask-derived state.
            if not apply_mask:
                features = features_for_pred

            # Predict split
            logits = self.split_heads[lvl](features_for_pred)
            gt_split = (octree_gt.children[d] >= 0).long()

            # Loss
            if apply_mask and mask.any():
                # standard CE on masked positions only
                loss_lvl = F.cross_entropy(logits[mask], gt_split[mask])
            elif self.use_focal_loss:
                loss_lvl = self.focal_loss(logits, gt_split)
            else:
                loss_lvl = F.cross_entropy(logits, gt_split)
            total_split_loss = total_split_loss + loss_lvl

            # Metrics: report both new (mask-only) and legacy (all-pos)
            with torch.no_grad():
                if n > 0:
                    acc_all_lvl = (
                        logits.argmax(-1) == gt_split).float().mean()
                else:
                    acc_all_lvl = torch.tensor(0.0, device=device)
                if mask.any():
                    acc_mask_lvl = (
                        logits[mask].argmax(-1) == gt_split[mask]
                    ).float().mean()
                else:
                    acc_mask_lvl = acc_all_lvl
                per_level_acc_all.append(acc_all_lvl)
                per_level_acc_mask.append(acc_mask_lvl)
                output[f'split_acc_lvl{lvl}'] = acc_mask_lvl

            # Expand using teacher-forced split
            split_mask_gt = (gt_split == 1)
            child_features = self._expand_features(features, split_mask_gt, lvl)
            child_features = child_features + self._get_pos_embed(
                octree_gt, d + 1)
            features = child_features

        # ---- 4. Leaf-level transformer ----
        nnum_leaf = octree_gt.nnum[self.depth_stop]
        assert features.shape[0] == nnum_leaf, "Final leaf count mismatch!"

        if features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree_gt, self.depth_stop)

        output["split_loss"] = total_split_loss / max(self.num_levels, 1)
        output["split_accuracy"] = (
            torch.stack(per_level_acc_mask).mean()
            if per_level_acc_mask else torch.tensor(0.0, device=device)
        )
        output["split_accuracy_all"] = (
            torch.stack(per_level_acc_all).mean()
            if per_level_acc_all else torch.tensor(0.0, device=device)
        )

        # ---- 5. VQ token prediction ----
        if gt_indices is not None and features.shape[0] > 0:
            vq_logits = self.vq_head(features)
            vq_logits_flat = vq_logits.reshape(-1, self.vq_size)
            gt_flat = gt_indices.reshape(-1).long()
            output["vq_loss"] = F.cross_entropy(vq_logits_flat, gt_flat)

            with torch.no_grad():
                pred_flat = vq_logits_flat.argmax(-1)
                output["vq_accuracy"] = (pred_flat == gt_flat).float().mean()
        else:
            output["vq_loss"] = torch.tensor(0.0, device=device)
            output["vq_accuracy"] = torch.tensor(0.0, device=device)

        output["loss"] = (self.split_weight * output["split_loss"]
                          + self.vq_weight * output["vq_loss"])
        return output

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch_size=1, device="cuda", temperature=0.8,
                 vqvae=None):
        """Generate with threshold-based split and temperature VQ sampling.

        Inference path: NO masking applied (mask_emb never touches features).
        The mid_transformer receives raw features; this is equivalent to the
        v4 inference path even when `use_masked_training=True`."""
        octree = ocnn.octree.init_octree(
            self.depth_stop, self.full_depth, batch_size, device)

        features = self.root_embedding.expand(
            octree.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree, self.full_depth)

        for lvl in range(self.num_levels):
            d = self.full_depth + lvl

            if features.shape[0] > 0:
                features = self._run_mid_transformer(
                    features, octree, d, lvl)

            logits = self.split_heads[lvl](features)
            probs = F.softmax(logits, dim=-1)
            split = (probs[:, 1] > self.split_threshold).long()

            octree.octree_split(split, d)
            octree.octree_grow(d + 1)

            features = self._expand_features(features, split.bool(), lvl)
            if features.shape[0] == 0:
                break

            features = features + self._get_pos_embed(octree, d + 1)

        if features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree, self.depth_stop)

        vq_logits = self.vq_head(features)
        vq_logits = vq_logits.reshape(-1, self.vq_groups, self.vq_size)

        if temperature > 0:
            probs = F.softmax(vq_logits / temperature, dim=-1)
            indices = torch.multinomial(
                probs.reshape(-1, self.vq_size), 1).reshape(-1, self.vq_groups)
        else:
            indices = vq_logits.argmax(-1)

        if vqvae is not None:
            vq_code = vqvae.quantizer.extract_code(indices)
        else:
            vq_code = indices.float()

        return octree, vq_code
