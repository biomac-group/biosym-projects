"""
Joint angles and ground reaction forces of the converged walking solutions, as full gait cycles.

    python scripts/extract_biosym_trials.py [SPEED | free] [--dir results/walking]

Reads ``<dir>/v<SPEED>/seed<k>.pkl`` (``paper_walking.py``) and writes ``<dir>/v<SPEED>/trials.npz`` (``free`` reads ``<dir>/free/``, the self-chosen speed runs) in the layout
of the BiomechPriorVAE evaluation rows: per converged trial the right hip/knee/ankle flexion [rad] and the right foot
force (Fx forward, Fy vertical) [body weights] over 100 nodes of one full gait cycle (the right leg's half cycle, then the
left leg's, which is the right leg half a cycle later), the pelvic obliquity, pelvis rotation and lumbar bending [rad], the duration of the full gait cycle and the walking speed (the set speed, or the one the optimiser chose).
"""
import argparse
import glob
import re
from pathlib import Path

import cloudpickle
import jax
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from biosym.constraints.dynamics import calc_forces
from biosym.objectives import state_prior_term as spt
from biosym.ocp import collocation

SPEED_TOL = 0.01      # m/s, closer than this to a speed bound counts as clamped
DUR_TOL = 0.005       # s, the same for the half-cycle duration
JOINTS = ("hip_flexion", "knee_angle", "ankle_angle")
LUMBAR_EXTENSION = "lumbar_extension"
FRONTAL = ("pelvis_obliquity", "pelvis_rotation", "lumbar_bending")   # the unpaired coordinates: mirrored with a sign flip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("speed", nargs="?", default="1.33", help="set speed [m/s] or \"free\"")
    parser.add_argument("--dir", default="results/walking")
    args = parser.parse_args()
    folder = Path(args.dir) / ("free" if args.speed == "free" else f"v{float(args.speed):.2f}")

    config = sorted(folder.glob("_config*.yaml"))[0]
    bounds = yaml.safe_load(config.read_text())["collocation"]["bounds"]
    speed_bounds = np.ravel(bounds["speed"]).astype(float)
    dur_bounds = np.ravel(bounds["dur"]).astype(float)
    ocp = collocation.Collocation(str(config))
    objective = [o for o in ocp.objective._objectives if isinstance(o, spt.Objective)][0]
    info, names = objective._get_info(), objective.prior["coord_names"]
    cols = info["cols"]
    index = {side: [names.index(f"{j}_{side}") for j in JOINTS] for side in "rl"}

    coordinates = [c.removeprefix("q_") for c in ocp.model.coordinates.names]
    frontal_idx = [coordinates.index(c) for c in FRONTAL]
    lumbar_idx = coordinates.index(LUMBAR_EXTENSION)
    angles, forces, durations, speeds, seeds, frontal, clamped, lumbar, objectives = [], [], [], [], [], [], [], [], []
    for path in sorted(glob.glob(str(folder / "seed*.pkl"))):
        (states, globals_), result, _ = cloudpickle.load(open(path, "rb"))
        if int(result["status"]) != 0:
            continue
        filled = jax.vmap(calc_forces, in_axes=(0, None, None))(
            states[: info["n_nodes"]], ocp.model, objective.obj_settings["constants"])
        q = np.asarray(filled.q)[:, cols]
        right = np.asarray(spt._foot_force(filled.ext_forces, info["force_idx"]["right"], info["body_weight"]))
        left = np.asarray(spt._foot_force(filled.ext_forces, info["force_idx"]["left"], info["body_weight"]))
        # right leg over the first step, the left leg (the right leg a step later) over the second
        angles.append(np.concatenate([q[:, index["r"]], q[:, index["l"]]]))
        forces.append(np.concatenate([right[:, [0, 1]], left[:, [0, 1]]]))
        half = np.asarray(filled.q)[:, frontal_idx]
        frontal.append(np.concatenate([half, -half]))        # the second step is the mirror image
        ext = np.asarray(filled.q)[:, lumbar_idx]
        lumbar.append(np.concatenate([ext, ext]))            # sagittal: the second step repeats the first
        objectives.append(float(result["obj_val"]))
        durations.append(2.0 * float(np.ravel(globals_.dur)[0]))
        speeds.append(float(np.linalg.norm(np.ravel(globals_.speed))))
        half = float(np.ravel(globals_.dur)[0])
        free_speed = len(speed_bounds) == 2 and speed_bounds[1] > speed_bounds[0]
        clamped.append(bool(half <= dur_bounds[0] + DUR_TOL or half >= dur_bounds[1] - DUR_TOL or
                            (free_speed and (speeds[-1] <= speed_bounds[0] + SPEED_TOL or
                                             speeds[-1] >= speed_bounds[1] - SPEED_TOL))))
        seeds.append(int(re.search(r"seed(\d+)", path).group(1)))

    angles, forces, durations = np.array(angles), np.array(forces), np.array(durations)
    # distance covered by the pelvis over the full cycle divided by its duration is the imposed speed
    out = folder / "trials.npz"
    np.savez(out, angles=angles, grf=forces, dur=durations, speed=np.array(speeds),
             seed=np.array(seeds), frontal=np.array(frontal), clamped=np.array(clamped), lumbar_ext=np.array(lumbar),
             objective=np.array(objectives))
    print(f"{out}: {len(seeds)} converged trials, {int(np.sum(clamped))} on a bound (seeds {[s for s, c in zip(seeds, clamped) if c]}), seeds {seeds}, cycle durations {np.round(durations, 3)}")


if __name__ == "__main__":
    main()
