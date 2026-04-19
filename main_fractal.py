"""
Training / generation entry-point for FractalGenerator (VQ-VAE version).

Usage:
  # Training
  python main_fractal.py --config configs/shapenet_fractal.yaml

  # Generation (after training)
  python main_fractal.py --config configs/shapenet_fractal.yaml \
      SOLVER.run generate SOLVER.ckpt <path_to_ckpt>
"""

import os
import sys
import copy

# ---- path setup ----
_ROOT = os.path.dirname(os.path.abspath(__file__))
_OCTGPT = os.path.normpath(os.path.join(_ROOT, "..", "octgpt"))
sys.path.insert(0, _OCTGPT)
sys.path.insert(0, _ROOT)

import torch
import ocnn
from tqdm import tqdm

from thsolver import Solver
from ognn.octreed import OctreeD
from octgpt.utils import utils, builder
from fractal_models.fractal_generator import FractalGenerator


class FractalSolver(Solver):
    """Solver that trains / evaluates / generates with FractalGenerator + VQ-VAE."""

    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.depth = FLAGS.MODEL.depth
        self.depth_stop = FLAGS.MODEL.depth_stop
        self.full_depth = FLAGS.MODEL.full_depth

    # ------------------------------------------------------------------
    # Model & dataset
    # ------------------------------------------------------------------

    def get_model(self, flags):
        # Build the fractal generator
        model = FractalGenerator(**flags.FractalGen)
        model.cuda(device=self.device)
        self.model_module = model

        # Build and freeze the pre-trained VQ-VAE
        vqvae = builder.build_vae_model(flags.VQVAE)
        vqvae.cuda(device=self.device)

        # Load pre-trained VQ-VAE weights
        vqvae_ckpt = flags.vqvae_ckpt
        checkpoint = torch.load(vqvae_ckpt, weights_only=True, map_location="cuda")
        vqvae.load_state_dict(checkpoint)
        print(f"Loaded frozen VQ-VAE from {vqvae_ckpt}")

        # Freeze VQ-VAE — never train it
        utils.set_requires_grad(vqvae, False)
        vqvae.eval()
        self.vqvae_module = vqvae

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
            vqvae=self.vqvae_module,
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
        if epoch % 20 != 0:
            return
        super().test_epoch(epoch)
        if self.is_master:
            self.generate_step(epoch)

    # ------------------------------------------------------------------
    # Generation: fractal expand → VQ tokens → VQ-VAE decode → mesh
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_step(self, index):
        """Generate one shape using fractal expansion + VQ-VAE decoding.

        Pipeline:
        1. FractalGenerator produces octree structure + VQ codes at depth_stop
        2. Extend octree from depth_stop to full depth (depth 8) with zero splits
        3. VQ-VAE decoder reconstructs SDF from codes
        4. Marching cubes extracts mesh
        """
        model = self.model_module
        vqvae = self.vqvae_module
        model.eval()

        with torch.autocast("cuda", enabled=self.use_amp):
            octree, vq_code = model.generate(
                batch_size=1, device=self.device,
                temperature=0.8, vqvae=vqvae)

        print(f"Generated octree with {octree.nnum} nodes and VQ code shape {vq_code.shape}")
        if vq_code.shape[0] == 0:
            return

        # ---- Extend octree from depth_stop to full depth ----
        # The VQ-VAE decoder expects an octree at full depth.
        # We add empty splits from depth_stop to depth (e.g. 6→8).
        for d in range(self.depth_stop, self.depth):
            split_zero = torch.zeros(
                octree.nnum[d], device=octree.device).long()
            octree.octree_split(split_zero, d)
            octree.octree_grow(d + 1)

        # ---- Decode with VQ-VAE ----
        doctree = OctreeD(octree)
        code_depth = self.depth_stop
        output = vqvae.decode_code(
            vq_code, code_depth, doctree,
            copy.deepcopy(doctree), update_octree=True)

        # ---- Extract mesh via marching cubes ----
        save_dir = os.path.join(self.logdir, "results")
        os.makedirs(save_dir, exist_ok=True)
        mesh_path = os.path.join(save_dir, f"{index}.obj")

        utils.create_mesh(
            output['neural_mpu'],
            mesh_path,
            size=self.FLAGS.SOLVER.resolution,
            level=0.002,
            clean=True,
            bbmin=-self.FLAGS.SOLVER.sdf_scale,
            bbmax=self.FLAGS.SOLVER.sdf_scale,
            mesh_scale=self.FLAGS.DATA.test.points_scale,
            save_sdf=False)

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