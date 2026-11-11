"""
One summary table per results folder, with the duty factor of every run.

    python scripts/summarize_results.py [FOLDER ...]       # default: every results/<variant>/<speed folder>

Combines the shard summaries (``summary_*.csv`` of ``paper_walking.py``: every attempt with its IPOPT status, objective
and solve time), the trials of ``extract_biosym_trials.py`` (``trials.npz``) and the metabolic costs of
``evaluate_metabolics.py`` (``metabolics.csv``, optional) into ``<folder>/summary.csv``, one row per attempted seed:

``seed``, ``status`` (IPOPT), ``converged``, ``objective``, ``solve_seconds``; for converged runs ``speed`` [m/s],
``step_duration`` [s, half the gait cycle], ``cadence`` [steps/s], ``duty_factor`` (fraction of the gait cycle in
stance, from the smoothed contact mask as in Figure 1), ``stance_time`` [s], ``peak_vgrf`` [BW], ``gait`` (Walking,
Running or Stiff Running), ``double_contact``, ``on_bound`` (speed or step duration on its bound), ``used`` (converged,
single contact, not on a bound: the runs the figures use), ``umberger`` and ``bhargava`` [J/kg/m].
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_figure1 as f1          # noqa: E402  (duty factor, heel-strike roll, screens)

FS = 100


def summarize(folder):
    """Write and return the summary table of one results folder."""
    folder = Path(folder)
    shards = [pd.read_csv(f) for f in sorted(glob.glob(str(folder / "summary_*.csv"))) if Path(f).stat().st_size > 0]
    attempts = pd.concat(shards).drop_duplicates("seed", keep="last").set_index("seed") if shards else pd.DataFrame()
    table = pd.DataFrame(index=sorted(attempts.index)) if len(attempts) else pd.DataFrame()
    if len(attempts):
        table["status"] = attempts["walking_status"].astype(int)
        table["converged"] = attempts["converged"].astype(int)
        table["objective"] = attempts["objective"]
        table["solve_seconds"] = attempts["walking_seconds"]

    trials = folder / "trials.npz"
    if trials.exists():
        t = np.load(trials)
        rows = {}
        for i, seed in enumerate(t["seed"]):
            step = float(t["dur"][i]) / 2.0
            grf = np.roll(t["grf"][i], -f1.heel_strike_start(t["grf"][i][:, 1]), axis=0)
            angles = np.roll(t["angles"][i], -f1.heel_strike_start(t["grf"][i][:, 1]), axis=0)
            duty = f1.duty_factor(grf[:, 1], step)
            double = not f1.gc.has_single_contact_phase(grf[:, 1], FS / step)
            bound = bool(t["clamped"][i]) if "clamped" in t.files else False
            rows[int(seed)] = {"speed": float(t["speed"][i]), "step_duration": step, "cadence": 1.0 / step,
                               "duty_factor": duty, "stance_time": duty * 2.0 * step, "peak_vgrf": float(grf[:, 1].max()),
                               "gait": f1.gait_class(grf[:, 1], angles), "double_contact": int(double),
                               "on_bound": int(bound), "used": int(not double and not bound)}
        extra = pd.DataFrame.from_dict(rows, orient="index")
        table = table.join(extra, how="outer") if len(table) else extra
    cost = folder / "metabolics.csv"
    if cost.exists():
        table = table.join(pd.read_csv(cost).set_index("seed")[["umberger", "bhargava"]], how="left")
    table.index.name = "seed"
    out = folder / "summary.csv"
    table.round(5).to_csv(out)
    used = table["used"].sum() if "used" in table else 0
    print(f"{out}: {len(table)} attempts, {int(used)} used"
          + (f", duty factor {table.loc[table['used'] == 1, 'duty_factor'].mean():.3f}" if used else ""))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*")
    args = parser.parse_args()
    folders = args.folders or sorted(str(p.parent) for p in (PROJECT / "results").glob("*/*/trials.npz")
                                     if p.parent.parent.name != "standing")
    for folder in folders:
        summarize(folder)


if __name__ == "__main__":
    main()
