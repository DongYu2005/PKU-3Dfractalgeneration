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
import logging

# ---- path setup ----
_ROOT = os.path.dirname(os.path.abspath(__file__))
_OCTGPT = os.path.normpath(os.path.join(_ROOT, "octgpt"))
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
        for key in ["octree", "octree_in", "octree_gt", "label",
                     "pos", "sdf", "grad", "weight", "occu", "color"]:
            if key in batch:
                batch[key] = batch[key].cuda()

    def model_forward(self, batch):
        self.batch_to_cuda(batch)
        output = self.model(
            octree_gt=batch["octree_gt"],
            vqvae=self.vqvae_module,
            label=batch.get("label"),
        )
        return output

    # ------------------------------------------------------------------
    # Train / test steps
    # ------------------------------------------------------------------

    def train_step(self, batch):
        output = self.model_forward(batch)
        return {"train/" + k: v for k, v in output.items()}

    def train_epoch(self, epoch):
        from thsolver.tracker import AverageTracker
        self.model.train()
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)

        flags = self.FLAGS.SOLVER
        avg_tracker = AverageTracker()
        rng = range(len(self.train_loader))
        oom_count = 0
        for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
            if flags.empty_cache > 0 and it % flags.empty_cache == 0:
                torch.cuda.empty_cache()

            batch = next(self.train_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch

            self.optimizer.zero_grad(flags.zero_grad_to_none)
            try:
                with torch.autocast('cuda', enabled=self.use_amp):
                    output = self.train_step(batch)
                    loss = output['train/loss']

                clip_grad = flags.clip_grad
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if clip_grad > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_grad)
                    self.optimizer.step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                oom_count += 1
                logging.warning(
                    "OOM at epoch %d iter %d (total OOM: %d), skipping batch",
                    epoch, it, oom_count)
                continue

            avg_tracker.update(output)
            avg_tracker.record_time()

            log_per_iter = flags.log_per_iter
            if self.is_master and log_per_iter > 0 and it % log_per_iter == 0:
                notes = 'iter: %d' % it
                avg_tracker.log(epoch, msg_tag='- ', notes=notes,
                                print_time=False)

        if self.world_size > 1:
            avg_tracker.average_all_gather()
        if self.is_master:
            avg_tracker.log(epoch, self.summary_writer, print_time=True)
            if oom_count > 0:
                logging.info("Epoch %d: %d OOM batches skipped", epoch,
                             oom_count)

    def test_step(self, batch):
        with torch.no_grad():
            output = self.model_forward(batch)
        return {"test/" + k: v for k, v in output.items()}

    def test_epoch(self, epoch):
        # Called by the solver only at test_every_epoch boundaries. Generate a
        # sample set every 10 epochs (generation writes big .obj files; keep it
        # rarer than testing/checkpointing to save disk on the shared FS).
        gen_every = self.FLAGS.SOLVER.get("gen_every_epoch", 10)
        super().test_epoch(epoch)
        if self.is_master and (gen_every <= 0 or epoch % gen_every == 0):
            self.generate_step(epoch)

    # ------------------------------------------------------------------
    # Generation: fractal expand → VQ tokens → VQ-VAE decode → mesh
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_and_save(self, octree, vq_code, mesh_path):
        """Extend octree to full depth, VQ-VAE decode, marching cubes to obj."""
        if vq_code.shape[0] == 0:
            print(f"Empty octree, skipping {mesh_path}")
            return
        # alignment sanity: one code per depth_stop leaf, code dim = VQ embed dim
        assert vq_code.shape[0] == octree.nnum[self.depth_stop], (
            f"vq_code rows {vq_code.shape[0]} != depth_stop leaves "
            f"{octree.nnum[self.depth_stop]}")

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
        output = self.vqvae_module.decode_code(
            vq_code, code_depth, doctree,
            copy.deepcopy(doctree), update_octree=True)

        # ---- Extract mesh via marching cubes ----
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

    @torch.no_grad()
    def generate_step(self, index):
        """Generate shapes using fractal expansion + VQ-VAE decoding.

        Unconditional: one shape -> ``{index}.obj``.
        Class-conditional: one shape per class -> ``{index}_cls{c}.obj``, so the
        per-class diversity is directly visible in the output dir."""
        model = self.model_module
        vqvae = self.vqvae_module
        model.eval()
        save_dir = os.path.join(self.logdir, "results")
        os.makedirs(save_dir, exist_ok=True)

        if getattr(model, "use_class_cond", False):
            for c in range(model.num_classes):
                label = torch.full((1,), c, dtype=torch.long,
                                   device=self.device)
                with torch.autocast("cuda", enabled=self.use_amp):
                    octree, vq_code = model.generate(
                        batch_size=1, device=self.device,
                        temperature=0.8, vqvae=vqvae, label=label)
                print(f"[cls {c}] octree {octree.nnum} vq_code {vq_code.shape}")
                self._decode_and_save(
                    octree, vq_code,
                    os.path.join(save_dir, f"{index}_cls{c}.obj"))
        else:
            with torch.autocast("cuda", enabled=self.use_amp):
                octree, vq_code = model.generate(
                    batch_size=1, device=self.device,
                    temperature=0.8, vqvae=vqvae)
            print(f"Generated octree {octree.nnum} vq_code {vq_code.shape}")
            self._decode_and_save(
                octree, vq_code, os.path.join(save_dir, f"{index}.obj"))

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

        num_meshes = self.FLAGS.get("num_generate", 30)
        for i in tqdm(range(num_meshes), ncols=80):
            self.generate_step(i)


if __name__ == "__main__":
    FractalSolver.main()