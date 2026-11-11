"""
Copy the reference data and baseline results behind Figure 1 of the BiomechPriorVAE paper into ``data/``.

    python scripts/extract_paper_data.py [--repo ~/PhD/01_Python_Projects/biomechpriorVAE]

Reads the (read-only) BiomechPriorVAE repository and writes

* ``data/reference/``  the walking and running reference only (consolidated 0.6-1.6 m/s means and variances, the Falisse
  1.33 m/s reference, the Fukuchi 2.5/3.5/4.5 m/s means, variances and per-subject files, and the Abe et al. 2019 walking and running
  cost of transport behind Figure 4a);
* ``data/baselines/predsim_ensemble.npz``  every PredSim ensemble run (hip/knee/ankle [rad], Fx/Fy [BW] over 100
  nodes starting at heel strike, speed, duration, Bhargava cost, gait class, single-contact flag, duty factor);
* ``data/baselines/<source>_setspeed.csv``  per-trial speed, duty factor and cadence of GaitDynamics, GaitNet and
  GaitEncoder, the numbers Figure 1b/c plots for them.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE_FILES = ["gait_speed_mean_var.npz", "Falisse_133_noMet.mat", "Fukuchi_25_mean.mat", "Fukuchi_25_var.mat",
                   "Fukuchi_35_mean.mat", "Fukuchi_35_var.mat", "Fukuchi_45_mean.mat", "Fukuchi_45_var.mat",
                   "peerj-07-8290-s002.csv", "peerj-07-8290-s003.csv"]
REFERENCE_DIRS = ["fukuchi_processed"]
FS = 100        # nodes per gait cycle of every stored trial


def duty_factor(vgrf, dur, gc):
    """Fraction of the cycle in stance, from a vertical GRF trace [BW]."""
    fs = FS / dur if dur and np.isfinite(dur) and dur > 0 else FS
    return float(np.mean(gc.contact_mask(np.asarray(vgrf, dtype=float), fs)))


def cadence(row):
    """Steps per second from the cycle duration (a stride holds two steps)."""
    dur = row.get("dur", np.nan)
    if not dur or not np.isfinite(dur) or dur <= 0:
        return np.nan
    return 2.0 / dur


def copy_reference(repo, data):
    target = data / "reference"
    target.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_FILES:
        shutil.copy2(repo / "reference_data" / name, target / name)
    for name in REFERENCE_DIRS:
        shutil.copytree(repo / "reference_data" / name, target / name, dirs_exist_ok=True)
    print(f"reference: {len(REFERENCE_FILES)} files + {REFERENCE_DIRS} -> {target}")


def write_predsim(gl, gm, gc, data):
    df = gl.load_baseline_ensemble()
    rows = {k: [] for k in ("angles", "grf", "speed", "dur", "cot", "gait", "single_contact", "duty", "rep")}
    for _, r in df.iterrows():
        rows["angles"].append(np.array([r["angles"][j] for j in ("hip_flexion", "knee_angle", "ankle_angle")]).T)
        rows["grf"].append(np.array([r["grf"][g] for g in ("grf_x", "grf_y")]).T)
        dur = float(r["dur"])
        fs = FS / dur if dur > 0 else FS
        rows["speed"].append(float(r["speed"]))
        rows["dur"].append(dur)
        rows["cot"].append(float(r["metabolicCost"]))
        rows["gait"].append(gm.classify_gait(r))
        rows["single_contact"].append(bool(gc.has_single_contact_phase(r["grf"]["grf_y"], fs)))
        rows["duty"].append(duty_factor(r["grf"]["grf_y"], dur, gc))
        rows["rep"].append(str(r["rep"]))
    target = data / "baselines"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "predsim_ensemble.npz", **{k: np.array(v) for k, v in rows.items()})
    print(f"predsim: {len(df)} runs ({sum(rows['single_contact'])} with a single contact phase) -> {target}")


def write_genbench(gl, gc, data):
    sets = {"gaitdynamics": lambda: gl.load_gaitdynamics(mode="vposef22"), "gaitnet": gl.load_gaitnet,
            "gaitencoder": gl.load_gaitencoder}
    target = data / "baselines"
    for name, load in sets.items():
        df = load()
        out = pd.DataFrame({"speed": df["speed"].astype(float),
                            "cadence_steps_per_s": df.apply(cadence, axis=1)})
        out["duty_factor"] = df["duty_factor"].astype(float) if "duty_factor" in df.columns else np.nan
        out.to_csv(target / f"{name}_setspeed.csv", index=False)
        print(f"{name}: {len(out)} trials, duty factor {'yes' if out['duty_factor'].notna().any() else 'no'}"
              f" -> {target / (name + '_setspeed.csv')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="~/PhD/01_Python_Projects/biomechpriorVAE")
    parser.add_argument("--data-dir", default=str(PROJECT / "data"))
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data = Path(args.data_dir)

    sys.path.insert(0, str(repo / "evaluation"))
    import os
    os.chdir(repo)                       # the loaders use repo-relative paths; nothing is written there
    import gait_contact as gc
    import gait_loading as gl
    import gait_metrics as gm

    copy_reference(repo, data)
    write_predsim(gl, gm, gc, data)
    write_genbench(gl, gc, data)


if __name__ == "__main__":
    main()
