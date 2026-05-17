"""Quick 3-view render of a generated .obj (front / 3-quarter / top).

Usage:
    python render_obj.py path/to/mesh.obj               # writes mesh.png next to it
    python render_obj.py path/to/mesh.obj out.png       # custom output path
"""
import sys
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


MAX_FACES = 150_000  # matplotlib gets slow past this


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    obj_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else obj_path.with_suffix(".png")

    print(f"Loading {obj_path} ...")
    mesh = trimesh.load(obj_path, process=False, force="mesh")
    print(f"  vertices={len(mesh.vertices)}  faces={len(mesh.faces)}")

    if len(mesh.faces) > MAX_FACES:
        try:
            mesh = mesh.simplify_quadric_decimation(MAX_FACES)
            print(f"  simplified: vertices={len(mesh.vertices)}  faces={len(mesh.faces)}")
        except Exception as e:
            print(f"  simplify failed ({e.__class__.__name__}: {e}); random face subsample")
            idx = np.random.choice(len(mesh.faces), MAX_FACES, replace=False)
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices, faces=mesh.faces[idx], process=False)

    v = mesh.vertices
    triangles = v[mesh.faces]
    center = (v.max(0) + v.min(0)) / 2
    extent = (v.max(0) - v.min(0)).max()
    triangles = (triangles - center) / extent

    light_dir = np.array([0.3, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shading = np.clip(mesh.face_normals @ light_dir, 0, 1)
    shading = 0.25 + 0.75 * shading
    colors = np.stack([shading] * 3 + [np.ones_like(shading)], axis=-1)

    views = [("front", (10, -90)), ("3/4", (25, -55)), ("top", (85, -90))]

    fig = plt.figure(figsize=(15, 5), dpi=150)
    for i, (name, (elev, azim)) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolors=colors, edgecolors="none", linewidths=0))
        lim = 0.55
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
