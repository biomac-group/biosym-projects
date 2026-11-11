"""
Figure 2 of the BiomechPriorVAE paper with the biosym solutions: sagittal-plane poses through one gait cycle.

    python scripts/plot_figure2.py [--results results/walking] [--out results/figure2_biosym.png]

One row per speed, eight columns for the gait-cycle phases 0, 12.5, ..., 87.5 % (phase 0 is the right heel strike). The
rows are the speeds biosym was run at: 0.8, 1.2 and 1.6 m/s walking, 2.0 m/s (the self-chosen-speed run that chose to
run), and 2.5, 3.5 and 4.5 m/s running. Each row shows one representative trial, the converged one with the lowest seed
that is free of double heel strikes and of the gait class of the row. The model is drawn from its bone meshes (shaded by
the surface orientation, painter's algorithm) by forward kinematics of the solved coordinates; muscles and ground
reaction arrows of the paper's figure are not drawn.
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_figure1 as f1          # noqa: E402  (trial selection, heel-strike roll, screen)

from biosym.model.model import BiosymModel                                       # noqa: E402
from biosym.results import viewer3d                                              # noqa: E402

# (target speed [m/s], results folder, gait classes of the row)
ROWS = ((0.8, "v0.80", ("Walking",)), (1.2, "v1.20", ("Walking",)), (1.6, "v1.60", ("Walking",)),
        (2.0, "free", ("Running", "Stiff Running")), (2.5, "v2.50", ("Running", "Stiff Running")),
        (3.5, "v3.50", ("Running", "Stiff Running")), (4.5, "v4.50", ("Running", "Stiff Running")))
PHASES = np.arange(8) * 12.5
FACE_STRIDE = 1         # draw every n-th triangle of the meshes
HALF_WIDTH = 1.15       # m shown to each side of the pelvis
HEIGHT = 2.0


def pick_trial(folder, classes, target):
    """Seed and speed of the representative trial of a folder: the lowest kept seed of the class (nearest to ``target``
    in the free-speed folder)."""
    ours = f1.load_ours(folder / "trials.npz")
    ok = np.flatnonzero(np.isin(ours["gait"], classes))
    if len(ok) == 0:
        raise SystemExit(f"{folder}: no trial of class {classes}")
    best = ok[np.argmin(np.abs(ours["speed"][ok] - target))] if folder.name == "free" else ok[np.argmin(ours["seed"][ok])]
    return int(ours["seed"][best]), float(ours["speed"][best])


def phase_poses(model, folder, seed):
    """Body names, rotations and origins at the eight phases of the cycle of one result, from the right heel strike."""
    saved = pickle.load(open(folder / f"seed{seed}.pkl", "rb"))
    q, _, _ = viewer3d.motion_from_solution(saved[0], model, 1, saved[2])
    q = q[:-1]                                              # the last frame repeats the first
    trial = np.load(folder / "trials.npz")
    grf = trial["grf"][list(trial["seed"]).index(seed)]
    q = np.roll(q, -f1.heel_strike_start(grf[:, 1]), axis=0)
    frames = np.round(PHASES / 100 * len(q)).astype(int) % len(q)
    return viewer3d.body_poses(model, q[frames])


def triangles(meshes, names, R, p):
    """Projected triangles (x, y), depth (z) and shading of one frame."""
    index = {n: i for i, n in enumerate(names)}
    polys, depth, shade = [], [], []
    for m in meshes:
        if m["body"] not in index:
            continue
        i = index[m["body"]]
        w = m["vertices"].astype(float) @ R[i].T + p[i]
        tri = w[m["faces"][::FACE_STRIDE]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        polys.append(tri[:, :, :2])
        depth.append(tri[:, :, 2].mean(axis=1))
        shade.append(np.abs(normal[:, 2]))
    return np.concatenate(polys), np.concatenate(depth), np.concatenate(shade)


def draw(ax, meshes, names, R, p, pelvis):
    polys, depth, shade = triangles(meshes, names, R, p)
    polys = polys - np.array([p[pelvis, 0], 0.0])           # centre on the pelvis
    order = np.argsort(depth)                               # far first, looking from the right side
    grey = 0.30 + 0.55 * shade[order]
    ax.add_collection(PolyCollection(polys[order], facecolors=np.stack([grey] * 3, axis=1), edgecolors=np.stack([grey] * 3, axis=1),
                                     linewidths=0.15, antialiaseds=False))
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
    ax.set_ylim(-0.05, HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(PROJECT / "results/walking"))
    parser.add_argument("--out", default=str(PROJECT / "results/figure2_biosym.png"))
    parser.add_argument("--rows", type=float, nargs="*", default=None, help="only these target speeds (testing)")
    args = parser.parse_args()
    results = Path(args.results)

    model = BiosymModel(str(PROJECT / "models/sipp_runmad/sipp_runmad.yaml"))
    folder = viewer3d.find_geometry_dir(model.osim_file)
    meshes = viewer3d.load_model_meshes(model.osim_file, folder)
    rows = [r for r in ROWS if args.rows is None or r[0] in args.rows]

    fig, axes = plt.subplots(len(rows), len(PHASES), figsize=(16, 2.1 * len(rows)), squeeze=False)
    for r, (target, name, classes) in enumerate(rows):
        seed, speed = pick_trial(results / name, classes, target)
        names, R, p = phase_poses(model, results / name, seed)
        pelvis = list(names).index("pelvis")
        print(f"{target} m/s: {name} seed {seed}, speed {speed:.2f} m/s", flush=True)
        for c in range(len(PHASES)):
            draw(axes[r, c], meshes, names, R[c], p[c], pelvis)
            if r == 0:
                axes[r, c].set_title(f"{PHASES[c]:g}%", fontsize=12)
        axes[r, 0].text(-0.02, 0.5, f"{speed:.2f} m/s", transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=12)
    fig.suptitle("Ours (biosym): sagittal-plane pose through the gait cycle (bone meshes)", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.005, wspace=0.0, hspace=0.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
