"""Render .obj meshes with OctGPT's pyrender-based renderer (EGL headless,
lit shading; much cleaner than the matplotlib fallback in render_obj.py).

Usage:
  python eval/render_octgpt.py <mesh.obj> [more.obj ...] [--views 5 7 0]
Writes <mesh>_octgpt.png next to each input (one image per view, then a
horizontal strip <mesh>_octgpt.png combining them).
"""
import os
import sys
import argparse

os.environ["PYOPENGL_PLATFORM"] = "egl"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "octgpt"))

import numpy as np
import trimesh
from PIL import Image
from metrics.render.render import render_mesh
from metrics.render_utils import scale_to_unit_sphere


def render_views(path, views):
    mesh = trimesh.load(path, force="mesh")
    mesh = scale_to_unit_sphere(mesh)
    panels = []
    for v in views:
        img = render_mesh(mesh, index=v, resolution=1024)
        panels.append(np.asarray(img, dtype=np.uint8))
    strip = np.concatenate(panels, axis=1)
    out = os.path.splitext(path)[0] + "_octgpt.png"
    Image.fromarray(strip).save(out)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--views", type=int, nargs="+", default=[5, 1, 16])
    a = ap.parse_args()
    for p in a.meshes:
        render_views(p, a.views)
