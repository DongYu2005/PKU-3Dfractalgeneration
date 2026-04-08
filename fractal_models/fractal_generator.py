"""
FractalGenerator: Fractal octree generation with VQ-VAE discrete token prediction.

Architecture:
  1. Learnable root embedding + spatial position encoding at full_depth
  2. Per-level split prediction + LocalCrossAttentionExpander for feature expansion
  3. Leaf-level OctFormer transformer for feature refinement
  4. VQ token prediction head (cross-entropy on BSQ indices)
  5. Pre-trained frozen VQ-VAE decoder for mesh extraction

Key design:
  - Fractal hierarchical expansion: depth 3 → depth_stop (e.g. 6)
  - Split + prune at each level (like FRACTAL3DGEN)
  - Leaf features → predict discrete VQ codes → VQ-VAE decoder → SDF → mesh
  - Position encoding at every level so nodes know WHERE they are in space
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm

# Ensure octgpt is importable
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
# Task 1: Transformer-based Feature Expansion (Local Cross-Attention)
# ============================================================================

class OctantPositionEmbedding(nn.Module):
    """Fixed 3D position embedding for the 8 octant children."""

    def __init__(self, dim: int):
        super().__init__()
        offsets = torch.tensor([
            [-1, -1, -1], [-1, -1, +1], [-1, +1, -1], [-1, +1, +1],
            [+1, -1, -1], [+1, -1, +1], [+1, +1, -1], [+1, +1, +1],
        ], dtype=torch.float32)
        self.register_buffer("offsets", offsets)
        self.proj = nn.Linear(3, dim)

    def forward(self) -> torch.Tensor:
        return self.proj(self.offsets)


class LocalCrossAttentionExpander(nn.Module):
    """Local cross-attention block: parent (N, C) -> children (N*8, C)."""

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

    def forward(self, parent_features: torch.Tensor) -> torch.Tensor:
        N = parent_features.shape[0]
        if N == 0:
            return torch.zeros(0, self.dim, device=parent_features.device)
        if parent_features.dim() == 2:
            context = parent_features.unsqueeze(1)
        else:
            context = parent_features
        q = self.octant_queries.unsqueeze(0).expand(N, -1, -1)
        q = q + self.octant_pos_emb().unsqueeze(0)
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            query=q_normed, key=kv_normed, value=kv_normed)
        children = q + attn_out
        children = children + self.ffn(self.norm_ffn(children))
        return children.reshape(N * 8, self.dim)


# ============================================================================
# FractalGenerator (main model — VQ-VAE version)
# ============================================================================

class FractalGenerator(nn.Module):
    """Fractal octree generator with VQ token prediction at leaf level.

    Training: split loss (CE) + VQ token loss (CE on BSQ indices)
    Inference: fractal expand → predict VQ indices → VQ-VAE decode → mesh
    """

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
        # VQ-VAE config (BSQ: 32 groups, each binary)
        vq_groups: int = 32,
        vq_size: int = 2,
        # Task 1: Expander attention heads
        expander_num_heads: int = 4,
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

        PosEmb = eval(pos_emb_type)
        Norm = eval(norm_type)

        # ---- learnable root features ----
        self.root_embedding = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.root_embedding, std=0.02)

        # ---- spatial position projection (xyz -> feature_dim) ----
        # Injects octree node coordinates so each node knows WHERE it is
        self.pos_proj = nn.Linear(3, feature_dim)

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

        # ---- per-level feature expanders (Task 1) ----
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

        # ---- VQ code projection ----
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
        """Extract normalized (x, y, z) coordinates and project to feature_dim.

        Each node gets a unique spatial embedding so the model knows
        WHERE in 3D space this node sits.
        """
        ox, oy, oz, ob = octree.xyzb(depth)
        scale = 2 ** depth
        pos = torch.stack([
            ox.float() / scale,
            oy.float() / scale,
            oz.float() / scale,
        ], dim=-1)  # (N, 3)
        return self.pos_proj(pos)  # (N, feature_dim)

    def _run_leaf_transformer(self, features, octree, depth):
        """Run the leaf-level OctFormer for feature refinement."""
        octreeT = OctreeT(
            octree, features.shape[0], self.patch_size, self.dilation,
            nempty=False, depth_list=[depth], buffer_size=0,
            use_swin=self.use_swin)
        feat = depth2batch(features, octreeT.indices)
        feat = self.leaf_transformer(feat, octreeT, context=None)
        feat = batch2depth(feat, octreeT.indices)
        return self.leaf_norm(feat)

    def _expand_features(self, features, split_mask, level_idx):
        """Expand split parents into 8 children via local cross-attention."""
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
        """Training forward pass with teacher forcing."""
        device = octree_gt.device
        output = {}

        # ---- 1. Extract GT VQ targets from frozen VQ-VAE ----
        if vqvae is not None:
            with torch.no_grad():
                vq_code = vqvae.extract_code(octree_gt)
                _, gt_indices, _ = vqvae.quantizer(vq_code)
        else:
            gt_indices = None

        # ---- 2. Initialise features at full_depth WITH position encoding ----
        features = self.root_embedding.expand(
            octree_gt.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree_gt, self.full_depth)

        total_split_loss = torch.tensor(0.0, device=device)
        total_split_acc = 0.0

        # ---- 3. Fractal expansion: full_depth -> depth_stop-1 ----
        for lvl in range(self.num_levels):
            d = self.full_depth + lvl
            nnum_d = octree_gt.nnum[d]

            assert features.shape[0] == nnum_d, \
                f"Shape mismatch at depth {d}: features={features.shape[0]}, nnum={nnum_d}"

            # Predict split (occupied / empty)
            logits = self.split_heads[lvl](features)
            gt_split = (octree_gt.children[d] >= 0).long()
            # 加权：惩罚漏分裂（false negative）比误分裂更严重
            split_ce_weight = torch.tensor([1.0, 3.0], device=device)
            total_split_loss = total_split_loss + F.cross_entropy(
                logits, gt_split, weight=split_ce_weight)

            with torch.no_grad():
                total_split_acc += (logits.argmax(-1) == gt_split).float().mean().item()

            # Teacher Forcing: only expand occupied nodes
            split_mask = (gt_split == 1)

            # Expand features for split children
            child_features = self._expand_features(features, split_mask, lvl)

            # Inject position encoding at the NEW depth level
            child_features = child_features + self._get_pos_embed(
                octree_gt, d + 1)

            features = child_features

        # ---- 4. Leaf-level transformer (depth_stop) ----
        nnum_leaf = octree_gt.nnum[self.depth_stop]
        assert features.shape[0] == nnum_leaf, "Final leaf count mismatch!"

        if features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree_gt, self.depth_stop)

        output["split_loss"] = total_split_loss / max(self.num_levels, 1)
        output["split_accuracy"] = torch.tensor(
            total_split_acc / max(self.num_levels, 1), device=device)

        # ---- 5. VQ token prediction loss ----
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

        # Total loss
        output["loss"] = (self.split_weight * output["split_loss"]
                          + self.vq_weight * output["vq_loss"])

        return output

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch_size=1, device="cuda", temperature=0.8,
                 vqvae=None):
        """Generate shapes by fractal expansion + VQ token sampling.

        Split uses argmax (greedy) for structural stability.
        VQ tokens use temperature sampling for diversity.
        """
        octree = ocnn.octree.init_octree(
            self.depth_stop, self.full_depth, batch_size, device)

        # Initialise features WITH position encoding
        features = self.root_embedding.expand(
            octree.nnum[self.full_depth], -1).contiguous()
        features = features + self._get_pos_embed(octree, self.full_depth)

        for lvl in range(self.num_levels):
            d = self.full_depth + lvl

            logits = self.split_heads[lvl](features)

            # GREEDY for split: structure must be stable, no random pruning!
            split = logits.argmax(-1)

            # Grow octree
            octree.octree_split(split, d)
            octree.octree_grow(d + 1)

            # Expand features
            features = self._expand_features(features, split.bool(), lvl)
            if features.shape[0] == 0:
                break

            # Inject position encoding at new depth
            features = features + self._get_pos_embed(octree, d + 1)

        # Leaf-level transformer
        if features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree, self.depth_stop)

        # Predict VQ tokens
        vq_logits = self.vq_head(features)
        vq_logits = vq_logits.reshape(-1, self.vq_groups, self.vq_size)

        # Temperature sampling for VQ tokens (diversity in surface detail)
        if temperature > 0:
            probs = F.softmax(vq_logits / temperature, dim=-1)
            indices = torch.multinomial(
                probs.reshape(-1, self.vq_size), 1).reshape(-1, self.vq_groups)
        else:
            indices = vq_logits.argmax(-1)

        # Convert indices to quantized codes
        if vqvae is not None:
            vq_code = vqvae.quantizer.extract_code(indices)
        else:
            vq_code = indices.float()

        return octree, vq_code