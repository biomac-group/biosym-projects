"""
Speed sweep of a model of this project (the set speeds of the BiomechPriorVAE paper), one speed per process.

    python scripts/sweep.py standing SEED [SEED ...]  [--model myofullbody] [--linear-solver mumps]
    python scripts/sweep.py walking SPEED --seeds 1 2 3  [--model myofullbody] [--objective metabolic|effort]
                                          [--max-iter 3000] [--linear-solver mumps]

``standing`` solves the quiet standing of each seed (a random pose on the prior, ``make_prior_guess``) to
``results/MODEL/standing_seed<k>.pkl``. ``walking`` solves the half cycle at the set speed from the standing solution of each
seed, with the problem of ``configs/MODEL_walking[_metabolic].yaml`` and the half-cycle duration bounds of the paper's
``running3D.m`` (``duration_bounds``: 0.2 s to ``1 / 2 / (0.27 v + 0.37)``, 0.5 above 2.5 m/s, 0.4 above 3 m/s). Results:
``results/MODEL/sweep/<objective>/v<speed>/seed<k>.pkl`` and, per speed, ``summary.csv`` (seed, status, objective terms,
half-cycle duration, seconds). A seed whose result exists is not repeated, so a stopped sweep resumes (a solve that was
interrupted by a restart has no result file and is solved again). An exception is written to ``seed<k>.failed.txt``, which
counts as done: a crash that repeats is not retried forever.

The paths of the configs are relative to the project folder, which this script changes into; the dashboard is off.
"""
import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from model_config import MODELS  # noqa: E402


def duration_bounds(speed):
    """Half-cycle duration bounds [s] of running3D.m for a set speed (``3d_simulations/scripts/paper_walking.py``)."""
    if speed > 3:
        return 0.2, 0.4
    if speed > 2.5:
        return 0.2, 0.5
    return 0.2, 1.0 / 2.0 / (0.27 * speed + 0.37)


def write_config(template, path, edit):
    """Copy a collocation yaml with the edits of ``edit(collocation_dict)`` to ``path``."""
    cfg = yaml.safe_load(open(template))
    edit(cfg["collocation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def common(c, linear_solver, max_iter):
    """Settings every sweep problem shares: no dashboard, the linear solver and the iteration limit."""
    s = c["settings"]
    s["enable_logging"] = False
    s.pop("dashboard_port", None)
    s.pop("iteration_log_interval", None)
    if linear_solver:
        s["linear_solver"] = linear_solver
    if max_iter:
        s["max_iter"] = int(max_iter)


def standing(model, seeds, linear_solver):
    """Quiet standing of the given seeds (skipped when the result exists)."""
    from biosym.ocp.collocation import Collocation
    from biosym.ocp.utils.prior_guess import make_prior_guess
    from biosym.ocp.utils.vectorize import states_dict_to_x
    from objectives.prior_map import PriorCoordinates

    cfg = MODELS[model]
    for seed in seeds:
        out = Path(f"results/{model}/standing_seed{seed}.pkl")
        if out.exists():
            print(f"standing seed {seed}: exists", flush=True)
            continue
        t = time.time()
        config = write_config(f"configs/{model}_standing.yaml", Path(f"results/{model}/sweep/_configs/standing_seed{seed}.yaml"),
                              lambda c: (common(c, linear_solver, None), c["settings"]["output"].update(file=str(out))))
        ocp = Collocation(str(config))
        guess = make_prior_guess(ocp.model, ocp.settings, ocp.initial_guess_states, seed=seed,
                                 coordinate_map=PriorCoordinates(ocp.model, cfg.thorax))
        ocp.x0 = states_dict_to_x(guess["states"], None)
        ocp.solve()
        print(f"standing seed {seed}: done in {time.time() - t:.0f} s", flush=True)


def walking(model, speed, seeds, objective, linear_solver, max_iter):
    """The walking half cycle at a set speed from the standing solution of each seed."""
    from biosym.ocp.collocation import Collocation

    folder = Path(f"results/{model}/sweep/{objective}/v{speed:.2f}")
    folder.mkdir(parents=True, exist_ok=True)
    template = f"configs/{model}_walking{'_metabolic' if objective == 'metabolic' else ''}.yaml"
    dur = duration_bounds(speed)
    for seed in seeds:
        result, failed = folder / f"seed{seed}.pkl", folder / f"seed{seed}.failed.txt"
        if result.exists() or failed.exists():
            print(f"v{speed:.2f} seed {seed}: done before", flush=True)
            continue
        start = Path(f"results/{model}/standing_seed{seed}.pkl")
        if not start.exists():
            standing(model, [seed], linear_solver)

        def edit(c):
            common(c, linear_solver, max_iter)
            c["name"] = f"{model}_{objective}_v{speed:.2f}_seed{seed}"
            c["settings"]["output"]["file"] = str(result)
            c["bounds"]["speed"] = float(speed)
            c["bounds"]["dur"] = [float(dur[0]), float(dur[1])]
            c["initial_guess"]["file"] = str(start)
        config = write_config(template, Path(f"results/{model}/sweep/_configs/{objective}_v{speed:.2f}_seed{seed}.yaml"), edit)
        t = time.time()
        try:
            solution = Collocation(str(config)).solve()
        except Exception:
            failed.write_text(traceback.format_exc())
            print(f"v{speed:.2f} seed {seed}: FAILED, see {failed}", flush=True)
            continue
        info = solution.info
        row = {"seed": seed, "speed": speed, "status": info.get("status"), "dur_half_cycle": float(solution.globals.dur),
               "seconds": round(time.time() - t), **{k: float(v) for k, v in solution.objective_values.items()}}
        summary = folder / "summary.csv"
        new = not summary.exists()
        with open(summary, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"v{speed:.2f} seed {seed}: status {row['status']}, {row['seconds']} s, {row}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("standing", "walking"))
    parser.add_argument("what", nargs="+", help="seeds (standing) or the set speed [m/s] (walking)")
    parser.add_argument("--model", default="myofullbody", choices=list(MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--objective", default="metabolic", choices=("metabolic", "effort"))
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--linear-solver", default=None)
    args = parser.parse_args()
    if args.stage == "standing":
        standing(args.model, [int(s) for s in args.what], args.linear_solver)
    else:
        walking(args.model, float(args.what[0]), args.seeds, args.objective, args.linear_solver, args.max_iter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
