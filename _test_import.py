"""Quick smoke test: import + instantiate FractalGenerator."""
import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'octgpt')))
sys.path.insert(0, os.path.dirname(__file__))

from fractal_models.fractal_generator import FractalGenerator
print('Import OK')

model = FractalGenerator(
    feature_dim=48, num_heads=4, blocks_per_level=2,
    full_depth=3, depth_stop=4, patch_size=512,
    dilation=2, drop_rate=0.0, pos_emb_type='SinPosEmb',
    norm_type='LayerNorm', use_checkpoint=False, use_swin=True,
    sdf_hidden_dim=48, sdf_num_layers=3,
    sdf_weight=1.0, split_weight=1.0, expander_num_heads=2,
)
total = sum(p.numel() for p in model.parameters())
print(f'Model created: {total:,} params')
print(f'  num_levels = {model.num_levels}')
print(f'  feature_dim = {model.feature_dim}')

# Quick forward test with a dummy octree
import torch
import ocnn

octree = ocnn.octree.Octree(depth=8, full_depth=3)
# Build a minimal octree from random points
points = torch.rand(1000, 3) * 2 - 1  # [-1, 1]
normals = torch.randn(1000, 3)
normals = normals / normals.norm(dim=1, keepdim=True)
pts = ocnn.octree.Points(points, normals)
pts.clip(-1, 1)
octree.build_octree(pts)

# Run forward
model.eval()
with torch.no_grad():
    output = model(octree_gt=octree)

print(f'\nForward pass OK!')
for k, v in output.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.item():.4f}')
    else:
        print(f'  {k}: {v:.4f}')

# Test generation
octree_gen, leaf_feats = model.generate(batch_size=1, device='cpu')
print(f'\nGeneration OK!')
print(f'  leaf_feats shape: {leaf_feats.shape}')
print(f'  octree nnum at depth_stop: {octree_gen.nnum[4]}')

# Test get_leaf_bboxes with sdf_scale
bboxes = model.get_leaf_bboxes(octree_gen, depth=4, batch_id=0, sdf_scale=0.9)
print(f'  bboxes shape: {bboxes.shape}')
print(f'  bboxes range: [{bboxes.min().item():.3f}, {bboxes.max().item():.3f}]')

print('\n=== ALL TESTS PASSED ===')
