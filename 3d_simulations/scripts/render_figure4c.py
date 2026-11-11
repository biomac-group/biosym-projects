"""
High-resolution render of Figure 4c: nominal against bouncy (no contact dissipation) contact in running at 4.5 m/s.

    python scripts/render_figure4c.py [--results results] [--out results/figure4c_render.png] [--dpi 400]

Follows the render of ``plot/figure04a.py`` of the BiomechPriorVAE repository (``OVERLAY_TRIALS``, ``_select_overlay_row``,
``_build_overlay_pose_payload``, ``_composite_overlay_row``): the nominal contact (orange) and the contact without dissipation
(red) are overlaid, semi-transparent (alpha 0.5), in one row of five phases evenly spaced from touch-down to the end of the
longer of the two stance phases. Each model is the converged trial without a double contact with the lowest Umberger cost at
4.5 m/s. The stiff variant is not rendered, as in the paper. Each foot with a vertical force of at least 0.05 BW gets a ground
reaction arrow (0.35 m per body weight, a darker shade of the model's colour) that starts at the centre of pressure; the
centre of pressure is the mean of the foot's contact-sphere positions weighted by their penetration, because the solved
external forces are per body, not per sphere. Skeletons and arrows are semi-transparent and the PNG has a transparent
background. The model is drawn from its bone meshes by forward kinematics of the solved coordinates (as ``plot_figure2.py``)
instead of Blender.
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_figure1 as f1                                  # noqa: E402
import plot_figure2 as f2                                  # noqa: E402
from biosym.model.model import BiosymModel                 # noqa: E402
from biosym.results import viewer3d                        # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
SPEED = 4.5
ENTRIES = (("effort", "#D55E00"), ("nodamp", "#D62728"))      # render order: nominal, bouncy; the paper's colours
N_PHASES = 5
STANCE_THRESHOLD = 0.05                                      # BW, as the paper's ``_stance_end_idx``
ALPHA = 0.5                                                  # skeleton opacity (the paper uses 0.75)
ARROW_ALPHA, ARROW_DARKEN = 0.5, 0.5                         # arrow opacity and its darkening of the model colour
GRF_SCALE, FOOT_GRF_MIN = 0.35, 0.05                         # m per BW; BW, the paper's constants
HALF_WIDTH, TOP = 0.62, 2.00                                 # m: drawn half width around the pelvis and height above ground


def lowest_cost_seed(folder):
    """Seed of the kept (converged, single-contact) trial of a folder with the lowest Umberger cost."""
    kept = f1.load_ours(folder / "trials.npz")
    cost = pd.read_csv(folder / "metabolics.csv").set_index("seed")["umberger"]
    seeds = [int(s) for s in kept["seed"] if int(s) in cost.index]
    return min(seeds, key=lambda s: cost[s])


def stance_end(vgrf):
    """First node after touch-down at node 0 where the vertical force falls below the threshold."""
    below = vgrf[1:] <= STANCE_THRESHOLD
    return int(np.argmax(below) + 1) if below.any() else len(vgrf)


def rolled(model, folder, seed):
    """Coordinates of the cycle of one result rolled to its heel strike, and its vertical force."""
    saved = pickle.load(open(folder / f"seed{seed}.pkl", "rb"))
    q, _, _ = viewer3d.motion_from_solution(saved[0], model, 1, saved[2])
    q = q[:-1]
    trial = np.load(folder / "trials.npz")
    vgrf = trial["grf"][list(trial["seed"]).index(seed)][:, 1]
    start = f1.heel_strike_start(vgrf)
    return np.roll(q, -start, axis=0), np.roll(trial["grf"][list(trial["seed"]).index(seed)], -start, axis=0)


def foot_arrows(spheres, names, R, p, grf, node):
    """(origin x, origin y, force x, force y) in the sagittal plane for each foot loaded at a node of the rolled cycle.

    The first half of ``grf`` is the right foot, the second the left foot half a cycle later, so the other foot's force is the
    same series shifted by 50 nodes.
    """
    arrows = []
    index = {n: i for i, n in enumerate(names)}
    for side, force in (("_r", grf[node]), ("_l", grf[(node + 50) % len(grf)])):
        if force[1] < FOOT_GRF_MIN:
            continue
        mine = [s for s in spheres if s["body"].endswith(side) and s["body"] in index]
        world = np.array([R[index[s["body"]]] @ s["pos"] + p[index[s["body"]]] for s in mine])
        depth = np.clip(np.array([s["radius"] for s in mine]) - world[:, 1], 0.0, None)
        weights = depth / depth.sum() if depth.sum() > 1e-9 else np.eye(len(mine))[np.argmin(world[:, 1] - [s["radius"] for s in mine])]
        cop = (weights[:, None] * world).sum(axis=0)
        arrows.append((cop[0], cop[1], force[0], force[1]))
    return arrows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(PROJECT / "results"))
    parser.add_argument("--out", default=str(PROJECT / "results/figure4c_render.png"))
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    results = Path(args.results)

    model = BiosymModel(str(PROJECT / "models/sipp_runmad/sipp_runmad.yaml"))
    meshes = viewer3d.load_model_meshes(model.osim_file, viewer3d.find_geometry_dir(model.osim_file))

    runs, grfs = [], []
    spheres = viewer3d.contact_spheres(model)
    for variant, color in ENTRIES:
        folder = results / variant / f"v{SPEED:.2f}"
        seed = lowest_cost_seed(folder)
        q, grf = rolled(model, folder, seed)
        runs.append((variant, color, seed, q, grf[:, 1]))
        grfs.append(grf)
        print(f"{variant}: seed {seed}, stance until node {stance_end(grf[:, 1])}", flush=True)
    nodes = np.round(np.linspace(0, max(stance_end(r[4]) for r in runs), N_PHASES)).astype(int)   # the same nodes for both

    poses = []
    for variant, color, seed, q, vgrf in runs:
        frames = np.round(nodes / len(vgrf) * len(q)).astype(int) % len(q)
        poses.append(viewer3d.body_poses(model, q[frames]))

    fig, axes = plt.subplots(1, N_PHASES, figsize=(1.45 * N_PHASES, 1.45 * (TOP + 0.03) / (2 * HALF_WIDTH)), squeeze=False)
    for k, ax in enumerate(axes[0]):
        centre = np.mean([p[k, list(names).index("pelvis"), 0] for names, R, p in poses])   # keeps the models' relative position
        for (variant, color, *_), (names, R, p) in zip(runs, poses):
            polys, depth, shade = f2.triangles(meshes, names, R[k], p[k])
            polys = polys - np.array([centre, 0.0])
            order = np.argsort(depth)
            tone = 0.35 + 0.6 * shade[order]
            rgb = np.stack([tone * c for c in to_rgb(color)], axis=1)
            ax.add_collection(PolyCollection(polys[order], facecolors=rgb, edgecolors="none", linewidths=0, alpha=ALPHA,
                                             antialiaseds=True))
        for (variant, color, *_), (names, R, p), grf in zip(runs, poses, grfs):
            dark = tuple(c * ARROW_DARKEN for c in to_rgb(color))
            for x0, y0, fx, fy in foot_arrows(spheres, names, R[k], p[k], grf, nodes[k]):
                ax.annotate("", xy=(x0 - centre + fx * GRF_SCALE, y0 + fy * GRF_SCALE), xytext=(x0 - centre, y0),
                            arrowprops=dict(arrowstyle="-|>", color=dark, alpha=ARROW_ALPHA, lw=2.0, mutation_scale=11,
                                            shrinkA=0, shrinkB=0), annotation_clip=False, zorder=5)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
        ax.set_ylim(-0.03, TOP)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, transparent=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
