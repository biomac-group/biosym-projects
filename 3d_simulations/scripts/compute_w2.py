"""
Per-speed 2-Wasserstein distances of Ours and the baselines to the reference (the numbers behind Figure 3).

    python scripts/compute_w2.py [--repo ~/PhD/01_Python_Projects/biomechpriorVAE] [--results results/walking]

For every speed of the sweep (0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5 m/s) and every model the ensemble of trials is
compared with the reference ensemble by ``gait_distributions.compare`` of the (read-only) BiomechPriorVAE repository:
hip/knee/ankle angles [deg] and Fx/Fy ground reaction forces [BW]. Ours are the converged biosym runs without double
heel strikes (``extract_biosym_trials.py``); GaitDynamics, GGN, GaitEncoder and the PredSim ensemble are loaded by the
repository's own loaders and filters. Writes ``data/w2/per_speed.csv`` with the columns ``model``, ``speed``,
``ang_w2``, ``grf_w2`` and ``n``.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SWEEP = (0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5)
FS = 100
GAIT_CLASS = {s: ("Walking",) if s <= 1.6 else ("Running", "Stiff Running") for s in SWEEP}


def ours_rows(results, gm):
    """Unified trial rows (as the repository stores them) of the biosym trials at the sweep speeds."""
    import plot_figure1 as f1

    rows = []
    for speed in SWEEP:
        folder = Path(results) / f"v{speed:.2f}"
        ours = f1.load_ours(folder / "trials.npz")
        if ours is None:
            continue
        for k in range(len(ours["seed"])):
            if ours["gait"][k] not in GAIT_CLASS[speed]:
                continue
            ang, grf = ours["angles"][k], ours["grf"][k]
            rows.append({"sweep_speed": speed, "dur": 2 * ours["step"][k], "seed": int(ours["seed"][k]),
                         "angles": {j: ang[:, i] for i, j in enumerate(gm.JOINT_COLS)},
                         "grf": {"grf_x": grf[:, 0], "grf_y": grf[:, 1]}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="~/PhD/01_Python_Projects/biomechpriorVAE")
    parser.add_argument("--results", default=str(PROJECT / "results/walking"))
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    results = Path(args.results).resolve()

    sys.path[:0] = [str(repo), str(repo / "evaluation")]
    os.chdir(repo)                       # the loaders use repo-relative paths; nothing is written there
    import gait_contact as gc
    import gait_distributions as gd
    import gait_loading as gl
    import gait_metrics as gm
    from compare_gaitdynamics_sweep import load_sources

    def converged(df):
        return df[df["converged"] == True] if "converged" in df.columns else df          # noqa: E712

    def single_contact(df):
        def keep(row):
            dur = row.get("dur", np.nan)
            fs = FS / dur if dur and np.isfinite(dur) and dur > 0 else FS
            return gc.has_single_contact_phase(row["grf"]["grf_y"], fs)
        return df[df.apply(keep, axis=1)] if len(df) else df

    sets = {"Ours": (ours_rows(results, gm), "sweep_speed")}
    by_label = {s.label: s for s in load_sources(labels=["gaitdynamics_vposef22", "gaitnet", "gaitencoder_prior"])}
    for model, label in (("GD", "gaitdynamics_vposef22"), ("GGN", "gaitnet"), ("GE", "gaitencoder_prior")):
        sets[model] = (by_label[label].df, by_label[label].speed_col)
    sets["PS"] = (single_contact(converged(gl.load_baseline_ensemble())), "v_target")

    out = []
    for model, (df, col) in sets.items():
        for speed in SWEEP:
            ens = df[np.isclose(df[col], speed)] if len(df) else df
            res = gd.compare(ens, speed, 8)
            if res:
                out.append({"model": model, "speed": speed, "ang_w2": res["ang_w2"], "grf_w2": res["grf_w2"],
                            "n": res["n_model"]})
    table = pd.DataFrame(out)
    target = PROJECT / "data" / "w2"
    target.mkdir(parents=True, exist_ok=True)
    table.to_csv(target / "per_speed.csv", index=False)
    print(table.pivot(index="speed", columns="model", values="ang_w2").round(2).to_string())
    print(table.pivot(index="speed", columns="model", values="grf_w2").round(3).to_string())
    print(f"wrote {target / 'per_speed.csv'}")


if __name__ == "__main__":
    main()
