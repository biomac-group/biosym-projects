"""
Cadence, duty factor and the error to the 1.2 and 1.4 m/s reference of the six walking runs (three models, two objectives).

    python scripts/gait_vs_reference.py

Per run (same definitions as the BiomechPriorVAE figure 1, ``3d_simulations/scripts/plot_figure1.py``):

* cadence: steps per minute, 60 / duration of a step (half the gait cycle);
* duty factor: fraction of the gait cycle with the right foot loaded (``plot_figure1.duty_factor``);
* the full gait cycle (right leg's half cycle, then the left leg's) of the right hip / knee / ankle angle [deg, gait3d sign:
  knee flexion negative] and the right foot's Fx (forward) and Fy (vertical) [body weights], rolled to the right heel strike,
  then circularly shifted (steps of a quarter frame) to the best waveform correlation of the ground reaction forces with the
  reference mean (``plot_figure1.best_shift``), once per reference speed.

The error is taken at the reference's major points: the prominent peaks and valleys of the reference mean curve of every
channel (prominence of at least 20 % of its range, circular). For each, the same kind of extremum of the run within +-12 %
of the cycle is found; the value error is ``|run - reference|`` there (degrees; body weights), and the MAE of a channel is
their mean. The reference with the lower mean range-normalised MAE over the five channels is "nearer". The reference has no
joint moments, so angles and ground reaction forces are what is compared.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import cloudpickle
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

PROJECTS = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECTS / "3d_simulations" / "scripts"
sys.path.insert(0, str(SCRIPTS / "figure_lib"))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du  # noqa: E402
import plot_figure1 as pf  # noqa: E402

from biosym.constraints.dynamics import calc_forces  # noqa: E402
from biosym.model.model import BiosymModel  # noqa: E402
from biosym.objectives.state_prior_term import Objective as PriorTerm, _foot_force  # noqa: E402
import render_compare as rc  # noqa: E402

REFERENCE_SPEEDS = (1.2, 1.4)
CHANNELS = ("hip", "knee", "ankle", "Fx", "Fy")
UNITS = ("deg", "deg", "deg", "BW", "BW")
WINDOW = 12            # frames (% of the cycle) searched around a reference extremum
PROMINENCE = 0.2       # of the range of the reference curve
SIPP = PROJECTS / "3d_simulations/models/sipp_runmad/sipp_runmad.yaml"
LARGE = PROJECTS / "large_models"
RUNS = [   # label, model yaml, result pickle, knee sign (the prior counts knee flexion negative)
    ("SIPP effort", SIPP, rc.best_sipp("effort", 1.3)[0], 1.0),
    ("SIPP metabolic", SIPP, rc.best_sipp("walking", 1.3)[0], 1.0),
    ("MS-Human effort", LARGE / "models/ms_human_700/ms_human_700.yaml", LARGE / "results/ms_human_700/walking_v1.30_seed1.pkl", -1.0),
    ("MS-Human metabolic", LARGE / "models/ms_human_700/ms_human_700.yaml", LARGE / "results/ms_human_700/walking_metabolic_v1.30_seed1.pkl", -1.0),
    ("MyoFullBody effort", LARGE / "models/myofullbody/myofullbody.yaml", LARGE / "results/myofullbody/walking_v1.30_seed1.pkl", -1.0),
    ("MyoFullBody metabolic", LARGE / "models/myofullbody/myofullbody.yaml", LARGE / "results/myofullbody/walking_metabolic_v1.30_seed1.pkl", -1.0),
]


def cycle(model, result, knee_sign):
    """
    Full gait cycle of a half-cycle solution, rolled to the right heel strike.

    Returns
    -------
        ``curves`` (100, 5): hip, knee, ankle [deg], Fx, Fy [BW] of the right leg; ``step`` [s].
    """
    (states, globals_), info, settings = cloudpickle.load(open(result, "rb"))
    n = int(settings["nnodes"])
    filled = jax.vmap(calc_forces, in_axes=(0, None, None))(states[:n], model, model.default_constants)
    names = [c.removeprefix("q_") for c in model.coordinates.names]
    q = np.asarray(filled.q)
    bw = float(model.reference_body_weight())
    force = {}
    for side, tag in (("r", "r"), ("l", "l")):
        idx = PriorTerm._side_indices(model.ext_forces.names, tag)
        force[side] = np.asarray(_foot_force(filled.ext_forces, idx, bw))[:, [0, 1]]
    sign = np.array([1.0, knee_sign, 1.0])
    angles = np.concatenate([np.stack([q[:, names.index(f"{j}_{s}")] for j in ("hip_flexion", "knee_angle", "ankle_angle")], axis=1) * sign
                             for s in "rl"])
    grf = np.concatenate([force["r"], force["l"]])
    start = pf.heel_strike_start(grf[:, 1])
    angles, grf = np.roll(angles, -start, axis=0), np.roll(grf, -start, axis=0)
    return np.concatenate([np.rad2deg(angles), grf], axis=1), float(np.ravel(globals_.dur)[0])


def extrema(curve):
    """Prominent peaks (+1) and valleys (-1) of a circular curve: list of ``(frame, kind)``."""
    n, span = len(curve), np.ptp(curve)
    ext = np.concatenate([curve, curve, curve])
    found = []
    for kind, y in ((1, ext), (-1, -ext)):
        p, _ = find_peaks(y, prominence=PROMINENCE * span)
        found += [(int(i - n), kind) for i in p if n <= i < 2 * n]
    return sorted(found)


def major_point_error(sim, ref):
    """MAE of the major points of one channel and the mean signed timing offset [% of the cycle]."""
    n = len(ref)
    errors, offsets = [], []
    for frame, kind in extrema(ref):
        window = np.arange(frame - WINDOW, frame + WINDOW + 1)
        values = sim[window % n]
        k = int(np.argmax(values) if kind > 0 else np.argmin(values))
        errors.append(abs(values[k] - ref[frame]))
        offsets.append(window[k] - frame)
    return float(np.mean(errors)), float(np.mean(offsets)), len(errors)


def main():
    refs = {}
    for speed in REFERENCE_SPEEDS:
        r = du.get_reference_data(speed, root_path=str(PROJECTS / "3d_simulations/data/reference"))
        refs[speed] = np.array([r["hip"]["mean"], r["knee"]["mean"], r["ankle"]["mean"], r["Fx"]["mean"], r["Fy"]["mean"]]).T
    duty_speeds, duty_ref, _ = pf.reference_duty()
    ref_duty = {s: float(d) for s, d in zip(duty_speeds, duty_ref)}
    print("reference mean curves: range per channel (hip knee ankle deg, Fx Fy BW):",
          {s: np.round(np.ptp(r, axis=0), 2).tolist() for s, r in refs.items()})
    print(f"reference duty factor: 1.2 m/s {ref_duty.get(1.2, float('nan')):.3f}, 1.4 m/s {ref_duty.get(1.4, float('nan')):.3f}\n")

    rows, models = [], {}
    for label, yaml, result, knee_sign in RUNS:
        if yaml not in models:
            models[yaml] = BiosymModel(str(yaml), compile_eom=False, fit_muscle_paths=False)
        sim, step = cycle(models[yaml], result, knee_sign)
        row = {"run": label, "step_s": step, "cadence_steps_per_min": 60.0 / step, "duty": pf.duty_factor(sim[:, 4], step)}
        best = None
        for speed, ref in refs.items():
            shift = pf.best_shift(sim, ref, channels=pf.ALIGN_CHANNELS)
            aligned = pf.roll_frac(sim, shift, axis=0)
            maes = [major_point_error(aligned[:, c], ref[:, c]) for c in range(5)]
            score = float(np.mean([m[0] / np.ptp(ref[:, c]) for c, m in enumerate(maes)]))
            row[f"score_{speed}"] = score
            row[f"shift_{speed}"] = shift
            for c, m in zip(CHANNELS, maes):
                row[f"mae_{c}_{speed}"], row[f"dt_{c}_{speed}"] = m[0], m[1]
            row[f"points_{speed}"] = sum(m[2] for m in maes)
            if best is None or score < best[1]:
                best = (speed, score)
        row["nearer"], row["nearer_score"] = best
        rows.append(row)

    print(f"{'run':24s} {'cad [steps/min]':>15s} {'duty':>6s} | nearer reference | " + "  ".join(f"{c} [{u}]" for c, u in zip(CHANNELS, UNITS))
          + " | normalised MAE 1.2 / 1.4")
    for r in rows:
        s = r["nearer"]
        print(f"{r['run']:24s} {r['cadence_steps_per_min']:15.1f} {r['duty']:6.3f} | {s:.1f} m/s          | "
              + "  ".join(f"{r[f'mae_{c}_{s}']:{'7.2f' if u == 'deg' else '7.3f'}}" for c, u in zip(CHANNELS, UNITS))
              + f" | {r['score_1.2']:.3f} / {r['score_1.4']:.3f}")
    out = LARGE / "results" / "gait_vs_reference.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    print("\nwritten:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
