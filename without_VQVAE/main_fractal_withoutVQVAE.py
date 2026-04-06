"""
Training / generation entry-point for FractalGenerator.

Usage:
  # Training
  python main_fractal.py --config configs/shapenet_fractal.yaml

  # Generation (after training)
  python main_fractal.py --config configs/shapenet_fractal.yaml \
      SOLVER.run generate SOLVER.ckpt <path_to_ckpt>
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys

# ---- path setup (must happen before any project / octgpt imports) ----
_ROOT = os.path.dirname(os.path.abspath(__file__))
_OCTGPT = os.path.normpath(os.path.join(_ROOT, "..", "octgpt"))
sys.path.insert(0, _OCTGPT)
sys.path.insert(0, _ROOT)

import torch
import ocnn
from tqdm import tqdm

from thsolver import Solver
from octgpt.utils import utils, builder
from fractal_models.fractal_generator import FractalGenerator


class FractalSolver(Solver):
    """Solver that trains / evaluates / generates with FractalGenerator."""

    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.depth = FLAGS.MODEL.depth
        self.depth_stop = FLAGS.MODEL.depth_stop
        self.full_depth = FLAGS.MODEL.full_depth

    # ------------------------------------------------------------------
    # Model & dataset
    # ------------------------------------------------------------------

    def get_model(self, flags):
        model = FractalGenerator(**flags.FractalGen)
        model.cuda(device=self.device)
        self.model_module = model
        return model

    def get_dataset(self, flags):
        return builder.build_dataset(flags)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def batch_to_cuda(self, batch):
        for key in ["octree", "octree_in", "octree_gt",
                     "pos", "sdf", "grad", "weight", "occu", "color"]:
            if key in batch:
                batch[key] = batch[key].cuda()

    def model_forward(self, batch):
        self.batch_to_cuda(batch)
        output = self.model(
            octree_gt=batch["octree_gt"],
            pos=batch.get("pos"),
            sdf=batch.get("sdf"),
            grad=batch.get("grad"),
        )
        return output

    # ------------------------------------------------------------------
    # Train / test steps
    # ------------------------------------------------------------------

    def train_step(self, batch):
        output = self.model_forward(batch)
        return {"train/" + k: v for k, v in output.items()}

    def test_step(self, batch):
        with torch.no_grad():
            output = self.model_forward(batch)
        return {"test/" + k: v for k, v in output.items()}

    def test_epoch(self, epoch):
        if epoch % 5 != 0:
            return
        super().test_epoch(epoch)
        if self.is_master:
            self.generate_step(epoch)

    # ------------------------------------------------------------------
    # Task 4: Sparse Octree-Guided Marching Cubes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_step(self, index):
        """Generate one shape using sparse SDF sampling guided by the octree.

        Instead of querying all 256^3 = 16M points, we:
        1. Get leaf bounding boxes from the generated octree.
        2. Only query SDF within occupied voxel regions (with padding).
        3. Fill the rest with default positive SDF (= outside the shape).
        This dramatically reduces the number of MLP forward calls.
        """
        model = self.model_module
        model.eval()

        with torch.autocast("cuda", enabled=self.use_amp):
            octree, leaf_feats = model.generate(
                batch_size=1, device=self.device)

        if leaf_feats.shape[0] == 0:
            return

        resolution = self.FLAGS.SOLVER.get("resolution", 256)
        sdf_scale = self.FLAGS.SOLVER.get("sdf_scale", 0.9)
        points_scale = self.FLAGS.DATA.test.get("points_scale", 0.5)
        size = resolution

        # ---- Task 4: Sparse sampling via octree leaf bboxes ----
        bboxes = model.get_leaf_bboxes(
            octree, self.depth_stop, batch_id=0,
            sdf_scale=sdf_scale)  # (N_leaves, 6), clamped to [-sdf_scale, sdf_scale]

        if bboxes.shape[0] == 0:
            return

        # Compute which grid cells are covered by leaf voxels
        # Map bboxes from [-sdf_scale, sdf_scale] to grid indices [0, size-1]
        def world_to_grid(w):
            return ((w + sdf_scale) / (2 * sdf_scale) * size).long().clamp(0, size - 1)

        # Padding in grid cells (1-2 cells for continuity at boundaries)
        pad = max(1, size // 128)

        # Build a boolean occupancy mask on the grid
        occ_mask = torch.zeros(size, size, size, dtype=torch.bool)

        bbox_np = bboxes.cpu()
        for i in range(bbox_np.shape[0]):
            x0 = max(0, world_to_grid(bbox_np[i, 0]).item() - pad)
            y0 = max(0, world_to_grid(bbox_np[i, 1]).item() - pad)
            z0 = max(0, world_to_grid(bbox_np[i, 2]).item() - pad)
            x1 = min(size - 1, world_to_grid(bbox_np[i, 3]).item() + pad)
            y1 = min(size - 1, world_to_grid(bbox_np[i, 4]).item() + pad)
            z1 = min(size - 1, world_to_grid(bbox_np[i, 5]).item() + pad)
            occ_mask[x0:x1+1, y0:y1+1, z0:z1+1] = True

        # If too many leaves, fallback to computing the mask more efficiently
        # via vectorized ops
        if bboxes.shape[0] > 10000:
            occ_mask = self._build_occ_mask_vectorized(
                bboxes, size, sdf_scale, pad)

        # Generate query points only for occupied cells
        occupied_indices = torch.nonzero(occ_mask, as_tuple=False)  # (K, 3)
        n_occupied = occupied_indices.shape[0]
        total_cells = size ** 3

        print(f"  Sparse MC: querying {n_occupied}/{total_cells} cells "
              f"({100*n_occupied/total_cells:.1f}%)")

        if n_occupied == 0:
            return

        # Convert grid indices to world coordinates
        grid_points = occupied_indices.float()
        world_points = grid_points / size * (2 * sdf_scale) - sdf_scale
        world_points = world_points.cuda()

        # Query SDF in batches
        max_batch = 64 ** 3
        sdf_list = []
        for head in range(0, world_points.shape[0], max_batch):
            pts = world_points[head: head + max_batch]
            with torch.autocast("cuda", enabled=self.use_amp):
                sdf_vals = model.eval_sdf(
                    leaf_feats, octree, self.depth_stop, pts, batch_id=0)
            sdf_list.append(sdf_vals.float().cpu())

        sdf_sparse = torch.cat(sdf_list)

        # Assemble into full 3D grid (default = 0.1 for empty space)
        sdf_vol = torch.full((size, size, size), 0.1)
        sdf_vol[occ_mask] = sdf_sparse
        sdf_np = sdf_vol.numpy()

        # ---- marching cubes ----
        vtx, faces, _ = utils.marching_cubes(sdf_np, level=0.002)
        if vtx.shape[0] == 0:
            return

        vtx = vtx * (2 * sdf_scale / size) - sdf_scale
        vtx = vtx * points_scale

        import trimesh
        save_dir = os.path.join(self.logdir, "results")
        os.makedirs(save_dir, exist_ok=True)
        mesh = trimesh.Trimesh(vtx, faces)
        mesh.export(os.path.join(save_dir, f"{index}.obj"))

    @staticmethod
    def _build_occ_mask_vectorized(bboxes, size, sdf_scale, pad):
        """Vectorized occupancy mask construction for many leaves."""
        occ_mask = torch.zeros(size, size, size, dtype=torch.bool)

        def w2g(w):
            return ((w + sdf_scale) / (2 * sdf_scale) * size).long().clamp(0, size - 1)

        # Process in chunks to avoid memory issues
        bbox_cpu = bboxes.cpu()
        x0s = (w2g(bbox_cpu[:, 0]) - pad).clamp(min=0)
        y0s = (w2g(bbox_cpu[:, 1]) - pad).clamp(min=0)
        z0s = (w2g(bbox_cpu[:, 2]) - pad).clamp(min=0)
        x1s = (w2g(bbox_cpu[:, 3]) + pad).clamp(max=size - 1)
        y1s = (w2g(bbox_cpu[:, 4]) + pad).clamp(max=size - 1)
        z1s = (w2g(bbox_cpu[:, 5]) + pad).clamp(max=size - 1)

        for i in range(len(bbox_cpu)):
            occ_mask[x0s[i]:x1s[i]+1, y0s[i]:y1s[i]+1, z0s[i]:z1s[i]+1] = True

        return occ_mask

    # ------------------------------------------------------------------
    # Bulk generation entry-point
    # ------------------------------------------------------------------

    def generate(self):
        """Generate many meshes (called via ``SOLVER.run generate``)."""
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.load_checkpoint()
        self.model.eval()

        num_meshes = self.FLAGS.get("num_generate", 100)
        for i in tqdm(range(num_meshes), ncols=80):
            self.generate_step(i)


if __name__ == "__main__":
    FractalSolver.main()
