"""
FractalGenerator: End-to-end 3D shape generation via fractal octree expansion.

Architecture:
  1. Learnable root embedding at full_depth
  2. Per-level split prediction + LocalCrossAttentionExpander for feature expansion
  3. Leaf-level OctFormer transformer for SDF feature refinement
  4. Local implicit decoder (MLP) for SDF prediction

Key innovations vs OctGPT:
  - No VQ-VAE: purely continuous features, fully differentiable
  - BFS parallel: all nodes at the same depth processed in one batch
  - End-to-end training: split loss + SDF loss
  - Transformer-based feature expansion: local cross-attention per parent
  - Sparse Marching Cubes: only query SDF in occupied octree regions

Changelog:
  Task 1: Replaced MLP feature_expanders with LocalCrossAttentionExpander
  Task 4: (in main_fractal.py) Sparse octree-guided Marching Cubes
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


# (Diffusion Loss removed — requires pretrained tokenizer GT latents.
#  Re-add as two-stage training once an octree autoencoder is available.)


# ============================================================================
# Task 1: Transformer-based Feature Expansion (Local Cross-Attention)
# ============================================================================

class OctantPositionEmbedding(nn.Module):
    """Fixed 3D position embedding for the 8 octant children.
    Each octant (0-7) maps to a corner offset in {-1, +1}^3.
    """

    def __init__(self, dim: int):
        super().__init__()
        # 8 octants: binary decomposition of index -> (x, y, z) offsets
        offsets = torch.tensor([
            [-1, -1, -1], [-1, -1, +1], [-1, +1, -1], [-1, +1, +1],
            [+1, -1, -1], [+1, -1, +1], [+1, +1, -1], [+1, +1, +1],
        ], dtype=torch.float32)  # (8, 3)
        self.register_buffer("offsets", offsets)
        self.proj = nn.Linear(3, dim)

    def forward(self) -> torch.Tensor:
        """Returns (8, dim) position embeddings."""
        return self.proj(self.offsets)


class LocalCrossAttentionExpander(nn.Module):
    """Replaces MLP feature_expander with a local cross-attention block.

    For each parent node:
      - Context (Key/Value): the parent feature [1, C]
        (Future extension: [K, C] where K includes neighboring parents)
      - Query: 8 learnable octant queries with 3D position embedding [8, C]
      - Attention is LOCAL per-parent: shape [N_parents, 8, C] — no global attn.

    Future extension interface:
      When context_features has shape [N_parents, K, C] (K > 1, e.g. K=4
      neighboring parents), the cross-attention naturally adapts because K/V
      length just changes from 1 to K. No architecture changes needed.
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Octant queries: 8 learnable tokens + 3D position embedding
        self.octant_queries = nn.Parameter(torch.zeros(8, dim))
        nn.init.normal_(self.octant_queries, std=0.02)
        self.octant_pos_emb = OctantPositionEmbedding(dim)

        # Cross-attention: Q from octant queries, K/V from parent context
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)

        # FFN after attention
        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, parent_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            parent_features: (N, C) — one feature per parent node.
              [Future: (N, K, C) for K context parents — just unsqueeze for K=1]

        Returns:
            children: (N*8, C) — 8 child features per parent, flattened.
        """
        N = parent_features.shape[0]
        if N == 0:
            return torch.zeros(0, self.dim, device=parent_features.device)

        # Context: (N, 1, C) — single parent as KV
        # [Future]: when context includes neighbors, shape becomes (N, K, C)
        # and this block works without changes.
        if parent_features.dim() == 2:
            context = parent_features.unsqueeze(1)  # (N, 1, C)
        else:
            context = parent_features               # already (N, K, C)

        # Queries: (8, C) -> (N, 8, C), broadcast for all parents
        q = self.octant_queries.unsqueeze(0).expand(N, -1, -1)  # (N, 8, C)
        q = q + self.octant_pos_emb().unsqueeze(0)              # add 3D pos

        # Cross-attention: Q attends to parent context
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            query=q_normed, key=kv_normed, value=kv_normed)
        children = q + attn_out  # residual

        # FFN
        children = children + self.ffn(self.norm_ffn(children))

        return children.reshape(N * 8, self.dim)  # (N*8, C)


# ============================================================================
# Local Implicit Decoder
# ============================================================================

class LocalImplicitDecoder(nn.Module):
    """Lightweight MLP that maps (feature, local_xyz) -> SDF value."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256,
                 num_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = feature_dim + 3
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, xyz], dim=-1)).squeeze(-1)


# ============================================================================
# FractalGenerator (main model)
# ============================================================================

class FractalGenerator(nn.Module):
    """Fractal octree generator with continuous features.

    OctFormer self-attention is only applied at the leaf level (depth_stop)
    for SDF feature refinement.  Intermediate levels rely solely on
    LocalCrossAttentionExpander for feature propagation.
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
        sdf_hidden_dim: int = 256,
        sdf_num_layers: int = 4,
        sdf_weight: float = 1.0,
        split_weight: float = 1.0,
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
        self.sdf_weight = sdf_weight
        self.split_weight = split_weight

        PosEmb = eval(pos_emb_type)
        Norm = eval(norm_type)

        # ---- learnable root features ----
        self.root_embedding = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.root_embedding, std=0.02)

        # ---- leaf-level transformer only (no per-level OctFormer) ----
        self.leaf_transformer = OctFormerStage(
            dim=feature_dim, num_heads=num_heads,
            num_blocks=blocks_per_level, patch_size=patch_size,
            dilation=dilation, attn_drop=drop_rate,
            proj_drop=drop_rate, dropout=drop_rate,
            nempty=False, use_checkpoint=use_checkpoint,
            use_swin=use_swin, pos_emb=PosEmb, norm_layer=Norm)
        self.leaf_norm = Norm(feature_dim)

        # ---- per-level split heads (depths full_depth … depth_stop-1) ----
        self.split_heads = nn.ModuleList([
            nn.Linear(feature_dim, 2) for _ in range(self.num_levels)
        ])

        # ---- Task 1: per-level Transformer-based feature expansion ----
        self.feature_expanders = nn.ModuleList([
            LocalCrossAttentionExpander(
                dim=feature_dim, num_heads=expander_num_heads)
            for _ in range(self.num_levels)
        ])

        # ---- shared child-position embedding (captures octant identity) ----
        self.child_pos_emb = nn.Parameter(torch.zeros(8, feature_dim))
        nn.init.normal_(self.child_pos_emb, std=0.02)

        # ---- local implicit decoder (SDF) ----
        self.sdf_decoder = LocalImplicitDecoder(
            feature_dim, sdf_hidden_dim, sdf_num_layers)

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
    # Transformer helpers
    # ------------------------------------------------------------------

    def _run_leaf_transformer(self, features: torch.Tensor, octree: Octree,
                              depth: int) -> torch.Tensor:
        """Run the leaf-level OctFormer for SDF feature refinement."""
        octreeT = OctreeT(
            octree, features.shape[0], self.patch_size, self.dilation,
            nempty=False, depth_list=[depth], buffer_size=0,
            use_swin=self.use_swin)
        feat = depth2batch(features, octreeT.indices)
        feat = self.leaf_transformer(feat, octreeT, context=None)
        feat = batch2depth(feat, octreeT.indices)
        return self.leaf_norm(feat)

    def _expand_features(self, features: torch.Tensor,
                         split_mask: torch.Tensor,
                         level_idx: int) -> torch.Tensor:
        """Task 1: Use LocalCrossAttentionExpander to predict 8 child features
        for every *split* parent node.

        Memory-safe: attention is LOCAL within each parent [N_parents, 8, C],
        never global across all 8N children.
        """
        parents = features[split_mask]  # (n_split, C)
        n = parents.shape[0]
        if n == 0:
            return torch.zeros(0, self.feature_dim, device=features.device)

        # Cross-attention expansion: (n, C) -> (n*8, C)
        children = self.feature_expanders[level_idx](parents)

        # Add shared octant positional bias
        children = children.view(n, 8, self.feature_dim)
        children = children + self.child_pos_emb.unsqueeze(0)
        return children.reshape(n * 8, self.feature_dim)

    def forward(self, octree_gt: ocnn.octree.Octree, pos=None, sdf=None, grad=None):
        """Training forward pass with teacher forcing.

        Args:
            octree_gt: Ground-truth octree (built from point cloud).
            pos:  (M, 4) SDF query points [x, y, z, batch_idx].
            sdf:  (M,)   Ground-truth SDF values.
            grad: (M, 3) Ground-truth SDF gradients (reserved, unused).
        """
        device = octree_gt.device
        output = {}

        # ---- 1. initialise features at full_depth ----
        # 初始特征数量必须严格等于真实八叉树顶层的节点数
        features = self.root_embedding.expand(
            octree_gt.nnum[self.full_depth], -1).contiguous()

        total_split_loss = torch.tensor(0.0, device=device)
        total_split_acc = torch.tensor(0.0, device=device)

        # ---- 2. fractal expansion: full_depth -> depth_stop-1 ----
        for lvl in range(self.num_levels):
            d = self.full_depth + lvl
            nnum_d = octree_gt.nnum[d]

            # 防御性检查：确保特征数量没乱，乱了说明网络结构设计有致命 Bug
            assert features.shape[0] == nnum_d, f"Shape mismatch at depth {d}: features={features.shape[0]}, nnum={nnum_d}"

            # 1) predict split (occupied / empty)
            logits = self.split_heads[lvl](features)
            
            # Ground Truth 分裂情况 (只有大于等于0的节点才是物理存在的实体子节点)
            gt_split = (octree_gt.children[d] >= 0).long()
            
            # 计算宏观结构的剪枝 Loss (CrossEntropy 或 BCE)
            total_split_loss = total_split_loss + F.cross_entropy(logits, gt_split)

            with torch.no_grad():
                total_split_acc += (logits.argmax(-1) == gt_split).float().mean()

            # 2) 绝对的 Teacher Forcing (保证树的拓扑结构与 octree_gt 严丝合缝)
            # 只有在真实的 3D 模型里被占据的节点，才向下展开特征！
            split_mask = (gt_split == 1)

            # 3) expand features for split children (Task 1: cross-attention)
            # 这一步将 N 个存活的父节点展开为 N*8 个子节点
            child_features = self._expand_features(features, split_mask, lvl)

            # 更新下一层的输入特征
            features = child_features

        # ---- 3. leaf-level transformer (depth_stop) ----
        # 到达目标层，此时生成的子节点数量必须等于八叉树里该层的物理节点数量
        nnum_leaf = octree_gt.nnum[self.depth_stop]
        assert features.shape[0] == nnum_leaf, "Final leaf count mismatch!"
        
        if features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree_gt, self.depth_stop)

        output["split_loss"] = total_split_loss / max(self.num_levels, 1)
        output["split_accuracy"] = total_split_acc / max(self.num_levels, 1)

        # ---- 4. SDF loss at leaf level ----
        if pos is not None and sdf is not None and features.shape[0] > 0:
            output["sdf_loss"] = self._sdf_loss(
                features, octree_gt, self.depth_stop, pos, sdf)
        else:
            output["sdf_loss"] = torch.tensor(0.0, device=device)

        # 汇总 Loss
        output["loss"] = (self.split_weight * output["split_loss"]
                            + self.sdf_weight * output["sdf_loss"])
                            
        return output

    # ------------------------------------------------------------------
    # SDF loss
    # ------------------------------------------------------------------

    def _sdf_loss(self, leaf_features: torch.Tensor, octree: Octree,
                  depth: int, pos: torch.Tensor,
                  sdf_gt: torch.Tensor) -> torch.Tensor:
        """Compute SDF L1 loss via leaf voxel lookup."""
        device = leaf_features.device
        scale = 2 ** depth

        ox, oy, oz, ob = octree.xyzb(depth)
        node_keys = (ob.long() * scale ** 3
                     + ox.long() * scale ** 2
                     + oy.long() * scale
                     + oz.long())
        sorted_keys, sort_idx = node_keys.sort()

        qxyz = pos[:, :3]
        qbatch = pos[:, 3].long()
        cell = ((qxyz + 1) / 2 * scale).long().clamp(0, scale - 1)
        q_keys = (qbatch * scale ** 3
                  + cell[:, 0] * scale ** 2
                  + cell[:, 1] * scale
                  + cell[:, 2])

        idx = torch.searchsorted(sorted_keys, q_keys).clamp(0, len(sorted_keys) - 1)
        matched = sort_idx[idx]
        valid = (node_keys[matched] == q_keys)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        feats = leaf_features[matched[valid]]
        cell_v = cell[valid].float()
        cell_size = 2.0 / scale
        cell_min = cell_v / scale * 2 - 1
        local = ((qxyz[valid] - cell_min) / cell_size * 2 - 1).clamp(-1, 1)

        return F.l1_loss(self.sdf_decoder(feats, local).float(), sdf_gt[valid])

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch_size: int = 1, device: str = "cuda",
                 temperature: float = 0.8):
        """Generate shapes by sequentially expanding the octree.

        Returns:
            octree:        The generated octree structure.
            leaf_features: (nnum[depth_stop], C) feature vectors at leaves.
        """
        octree = ocnn.octree.init_octree(
            self.depth_stop, self.full_depth, batch_size, device)
        features = self.root_embedding.expand(
            octree.nnum[self.full_depth], -1).contiguous()

        for lvl in range(self.num_levels):
            d = self.full_depth + lvl

            # sample split decisions (no transformer — just from features)
            logits = self.split_heads[lvl](features)
            if temperature > 0:
                split = torch.multinomial(
                    F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
            else:
                split = logits.argmax(-1)

            # grow octree
            octree.octree_split(split, d)
            octree.octree_grow(d + 1)

            # expand features via cross-attention
            features = self._expand_features(features, split.bool(), lvl)
            if features.shape[0] == 0:
                break

        # leaf-level transformer refinement for SDF
        if features.shape[0] > 0:
            features = self._run_leaf_transformer(
                features, octree, self.depth_stop)

        return octree, features

    # ------------------------------------------------------------------
    # SDF evaluation for mesh extraction
    # ------------------------------------------------------------------

    def eval_sdf(self, leaf_features: torch.Tensor, octree: Octree,
                 depth: int, points: torch.Tensor,
                 batch_id: int = 0) -> torch.Tensor:
        """Evaluate SDF at arbitrary query points (single-batch)."""
        device = leaf_features.device
        scale = 2 ** depth

        ox, oy, oz, ob = octree.xyzb(depth)
        mask_b = (ob == batch_id)
        node_keys = (ox[mask_b].long() * scale ** 2
                     + oy[mask_b].long() * scale
                     + oz[mask_b].long())
        node_feats = leaf_features[mask_b]
        sorted_keys, sort_idx = node_keys.sort()

        cell = ((points + 1) / 2 * scale).long().clamp(0, scale - 1)
        q_keys = cell[:, 0] * scale ** 2 + cell[:, 1] * scale + cell[:, 2]

        idx = torch.searchsorted(sorted_keys, q_keys).clamp(0, len(sorted_keys) - 1)
        matched = sort_idx[idx]
        valid = (node_keys[matched] == q_keys)

        sdf = torch.ones(points.shape[0], device=device) * 0.1
        if valid.sum() > 0:
            feats = node_feats[matched[valid]]
            cell_v = cell[valid].float()
            cell_size = 2.0 / scale
            cell_min = cell_v / scale * 2 - 1
            local = ((points[valid] - cell_min) / cell_size * 2 - 1).clamp(-1, 1)
            sdf[valid] = self.sdf_decoder(feats, local).float()
        return sdf

    # ------------------------------------------------------------------
    # Task 4: Get leaf bounding boxes for sparse Marching Cubes
    # ------------------------------------------------------------------

    def get_leaf_bboxes(self, octree: Octree, depth: int,
                        batch_id: int = 0,
                        sdf_scale: float = 1.0) -> torch.Tensor:
        """Return bounding boxes of occupied leaf voxels.

        Args:
            sdf_scale: Clamp bboxes to [-sdf_scale, sdf_scale] so they
                       align with the Marching Cubes query grid.

        Returns:
            bboxes: (N_leaves, 6) each row is [x_min, y_min, z_min,
                     x_max, y_max, z_max] clamped to [-sdf_scale, sdf_scale].
        """
        scale = 2 ** depth
        cell_size = 2.0 / scale

        ox, oy, oz, ob = octree.xyzb(depth)
        mask_b = (ob == batch_id)

        # Integer cell coordinates
        cx = ox[mask_b].float()
        cy = oy[mask_b].float()
        cz = oz[mask_b].float()

        # Convert to [-1, 1] world space, then clamp to query grid range
        x_min = (cx / scale * 2 - 1).clamp(-sdf_scale, sdf_scale)
        y_min = (cy / scale * 2 - 1).clamp(-sdf_scale, sdf_scale)
        z_min = (cz / scale * 2 - 1).clamp(-sdf_scale, sdf_scale)
        x_max = (x_min + cell_size).clamp(-sdf_scale, sdf_scale)
        y_max = (y_min + cell_size).clamp(-sdf_scale, sdf_scale)
        z_max = (z_min + cell_size).clamp(-sdf_scale, sdf_scale)

        bboxes = torch.stack([x_min, y_min, z_min,
                              x_max, y_max, z_max], dim=-1)
        return bboxes
