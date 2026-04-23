"""
FractalGenerator v4: Fractal octree generation with VQ-VAE token prediction.

Changes from v3:
  - Focal Loss for split prediction (handles class imbalance)
  - Per-level OctFormerStage (1 block) for node communication before split
  - Threshold-based split in generation (prob > 0.3 instead of argmax)
  - Position encoding at every level
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """
        Args:
            logits: (N, 2) raw predictions
            targets: (N,) with values in {0, 1}
        """
        probs = F.softmax(logits, dim=-1)
        # Gather the probability of the true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Per-sample alpha: alpha for positive (1), 1-alpha for negative (0)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal modulation: downweight easy samples
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        # Standard CE per sample
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
    """Parent (N, C) -> children (N*8, C) via local cross-attention."""

    def __init__(self, dim: int, num_heads: int = 4, ffn_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.octant_queries = nn.Parameter(torch.zeros(8, dim))
        nn.init.normal_(self.octant_queries, std=0.02)
        self.octant_pos_emb = OctantPositionEmbedding(dim)
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
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
        children = children + self.ffn(self.norm_ffn(children))
        return children.reshape(N * 8, self.dim)


# ============================================================================
# FractalGenerator v4
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

        PosEmb = eval(pos_emb_type)
        Norm = eval(norm_type)

        # ---- Focal Loss for split ----
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # ---- learnable root features ----
        self.root_embedding = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.root_embedding, std=0.02)

        # ---- spatial position projection ----
        self.pos_proj = nn.Linear(3, feature_dim)

        # ---- per-level mid transformers (node communication before split) ----
        # 1 block each — lightweight, just enough for neighbors to talk
        self.mid_transformers = nn.ModuleList([
            OctFormerStage(
                dim=feature_dim, num_heads=num_heads,
                num_blocks=1,
                patch_size=patch_size, dilation=dilation,
                attn_drop=drop_rate, proj_drop=drop_rate, dropout=drop_rate,
                nempty=False, use_checkpoint=use_checkpoint,
                use_swin=use_swin, pos_emb=PosEmb, norm_layer=Norm)
            for _ in range(self.num_levels)
        ])
        self.mid_norms = nn.ModuleList([
            Norm(feature_dim) for _ in range(self.num_levels)
        ])

        # ---- leaf-level transformer (6 blocks for final refinement) ----
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
                dim=feature_dim, num_heads=expander_num_heads)
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
        """1-block OctFormer for same-level node communication."""
        octreeT = OctreeT(
            octree, features.shape[0], self.patch_size, self.dilation,
            nempty=False, depth_list=[depth], buffer_size=0,
            use_swin=self.use_swin)
        feat = depth2batch(features, octreeT.indices)
        feat = self.mid_transformers[lvl](feat, octreeT, context=None)
        feat = batch2depth(feat, octreeT.indices)
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
        total_split_acc = 0.0

        # ---- 3. Fractal expansion ----
        for lvl in range(self.num_levels):
            d = self.full_depth + lvl
            nnum_d = octree_gt.nnum[d]

            assert features.shape[0] == nnum_d, \
                f"Shape mismatch at depth {d}: features={features.shape[0]}, nnum={nnum_d}"

            # Node communication: let neighbors talk before split decision
            if features.shape[0] > 0:
                features = self._run_mid_transformer(
                    features, octree_gt, d, lvl)

            # Predict split
            logits = self.split_heads[lvl](features)
            gt_split = (octree_gt.children[d] >= 0).long()

            # Focal Loss (handles class imbalance)
            total_split_loss = total_split_loss + self.focal_loss(logits, gt_split)

            with torch.no_grad():
                total_split_acc += (logits.argmax(-1) == gt_split).float().mean().item()

            # Teacher Forcing
            split_mask = (gt_split == 1)

            # Expand + inject position at new depth
            child_features = self._expand_features(features, split_mask, lvl)
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
        output["split_accuracy"] = torch.tensor(
            total_split_acc / max(self.num_levels, 1), device=device)

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
        """Generate with threshold-based split and temperature VQ sampling."""
        octree = ocnn.octree.init_octree(
            self.depth_stop, self.full_depth, batch_size, device)

        features = self.root_embedding.expand(
            octree.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree, self.full_depth)

        for lvl in range(self.num_levels):
            d = self.full_depth + lvl

            # Node communication before split
            if features.shape[0] > 0:
                features = self._run_mid_transformer(
                    features, octree, d, lvl)

            logits = self.split_heads[lvl](features)

            # Threshold-based split: if P(split) > threshold, split it
            # Much better than argmax (which collapses to all-0)
            # and more stable than multinomial sampling
            probs = F.softmax(logits, dim=-1)
            split = (probs[:, 1] > self.split_threshold).long()

            # Grow octree
            octree.octree_split(split, d)
            octree.octree_grow(d + 1)

            # Expand features
            features = self._expand_features(features, split.bool(), lvl)
            if features.shape[0] == 0:
                break

            # Position encoding at new depth
            features = features + self._get_pos_embed(octree, d + 1)

        # Leaf-level transformer
        if features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree, self.depth_stop)

        # Predict VQ tokens with temperature sampling
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