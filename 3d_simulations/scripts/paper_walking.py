"""
Walking of the BiomechPriorVAE paper pipeline: solve at one speed until a number of runs have converged.

    python scripts/paper_walking.py (SPEED | --free-speed LOW HIGH) [--n-converged 10] [--max-attempts 50] [--linear-solver mumps]
                                    [--nnodes 50] [--max-iter N] [--out-dir results]

Attempt ``k`` starts from the standing result of seed ``k`` (``results/standing/seed<k>.pkl``); when that file does
not exist ``paper_standing.py``'s pipeline is run with ``k`` as the seed. The walking problem of ``configs/walking.yaml``
(set speed, half cycle, Bhargava cost, state prior, arm effort, heel-strike variance penalty; running3D.m's duration bounds) is built once and every attempt solves it with
``nlp.solve(x0)``, as the standing problem is. An attempt counts as converged when IPOPT ends with "Optimal Solution
Found" (status 0). Results are ``results/walking/v<SPEED>/seed<k>.pkl``; the attempts are listed in
``results/walking/v<SPEED>/summary<TAG>.csv``. An attempt whose result file exists is not repeated, so a stopped run resumes.
"""
import argparse
import yaml
import csv
import sys
from pathlib import Path

import cloudpickle
import numpy as np

import paper_common as pc
from paper_standing import Standing


def speed_dir(out_dir, speed, variant="walking"):
    """Folder of all results of a variant at one set speed [m/s], or at a free (self-chosen) speed when ``speed`` is ``None``."""
    return Path(out_dir) / variant / ("free" if speed is None else f"v{speed:.2f}")


def walking_path(out_dir, speed, seed, variant="walking"):
    """Result file of an attempt."""
    return speed_dir(out_dir, speed, variant) / f"seed{seed}.pkl"


def duration_bounds(speed):
    """Half-cycle duration bounds [s] of running3D.m for a set speed: 0.2 s up to ``1 / 2 / (0.27 v + 0.37)``, 0.5 above
    2.5 m/s and 0.4 above 3 m/s."""
    if speed > 3:
        return 0.2, 0.4
    if speed > 2.5:
        return 0.2, 0.5
    return 0.2, 1.0 / 2.0 / (0.27 * speed + 0.37)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("speed", type=float, nargs="?", default=None,
                        help="set walking speed [m/s]; omit together with --free-speed")
    parser.add_argument("--free-speed", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                        help="optimise the speed within these bounds [m/s] (self-chosen speed) instead of setting it")
    parser.add_argument("--n-converged", type=int, default=10, help="stop after this many converged runs")
    parser.add_argument("--max-attempts", type=int, default=50, help="never try more seeds than this")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds to try, in order (default 1..max-attempts); for sharding one speed over processes")
    parser.add_argument("--tag", default="", help="suffix of the summary and config files, one per shard")
    parser.add_argument("--dur-max", type=float, default=None, help="upper bound of the half-cycle duration [s] (default: running3D.m's for the speed)")
    parser.add_argument("--dur-min", type=float, default=None, help="lower bound of the half-cycle duration [s] (default 0.2)")
    parser.add_argument("--variant", default="walking",
                        help="results folder below --out-dir (e.g. effort, weak30, stiff, nodamp)")
    parser.add_argument("--model", default=None, help="model definition (default: the config's standard model)")
    parser.add_argument("--cost", choices=("metabolic", "effort"), default="metabolic",
                        help="task objective: Bhargava cost of transport, or the cubed muscle effort of Nitschke et al.")
    parser.add_argument("--cot-arg", action="append", default=[], metavar="KEY=VALUE",
                        help="extra argument of the cot_term objective (repeatable), e.g. work=absolute or fast_twitch=1.0")
    parser.add_argument("--weight", action="append", default=[], metavar="TERM=VALUE",
                        help="objective weight of a term (repeatable), e.g. cot_term=0.5 state_prior_term=2")
    parser.add_argument("--nnodes", type=int, default=50, help="nodes of the half cycle")
    parser.add_argument("--linear-solver", default=None, help="IPOPT linear solver, e.g. ma97 or mumps")
    parser.add_argument("--max-iter", type=int, default=None, help="IPOPT iteration limit (default: the config's)")
    parser.add_argument("--log", action="store_true",
                        help="turn the iteration logger on (objective terms and constraints every 25 iterations, no dashboard)")
    parser.add_argument("--dashboard", action="store_true", help="with --log: also launch the live dashboard (port 8050)")
    parser.add_argument("--warm-from", default=None, metavar="DIR",
                        help="start every seed from the converged walking solution DIR/seed<k>.pkl (a nominal run) instead of "
                             "the standing solution; seeds without a converged file there are skipped")
    parser.add_argument("--out-dir", default=str(pc.RESULT_DIR))
    args = parser.parse_args()
    if (args.speed is None) == (args.free_speed is None):
        parser.error("give either a set SPEED or --free-speed LOW HIGH")
    speed = args.speed
    out_dir = Path(args.out_dir)
    log = pc.Stopwatch()

    warm = Path(args.warm_from) if args.warm_from else None
    standing = None if warm else Standing(out_dir, args.linear_solver)
    first_seed = args.seeds[0] if args.seeds else 1
    if standing:
        first = standing.run(first_seed)
        log(f"standing of seed {first_seed}: {'done before' if first['status'] is None else 'status %d' % first['status']}")
    guess_file = (lambda seed: warm / f"seed{seed}.pkl") if warm else standing.path

    def edit(cfg):
        cfg["bounds"]["speed"] = speed if args.free_speed is None else list(args.free_speed)
        cfg["settings"]["nnodes"] = args.nnodes
        cfg["settings"]["output"]["file"] = str(walking_path(out_dir, speed, 0, args.variant))
        cfg["initial_guess"] = {"type": "from_file", "file": str(guess_file(first_seed))}
        if args.model:
            cfg["settings"]["model"] = args.model
        if args.cost == "effort":
            # supplementary Table of the paper: muscle effort weight 1000 (cubed, volume weighted, divided by speed^3),
            # no metabolic cost, state prior weight 3; the arm torque effort keeps its weight
            terms = [o for o in cfg["objectives"] if o["name"] != "cot_term"]
            terms.insert(0, {"name": "effort_term", "weight": 1000,
                             "args": {"select": "muscles", "weighting": "volumeweighted", "exponent": 3,
                                      "speedweighting": True}})
            for o in terms:
                if o["name"] == "state_prior_term":
                    o["weight"] = 3
                elif o["name"] == "effort_term" and o["args"].get("select") == "torques":
                    o["weight"] = 1
            cfg["objectives"] = terms
        for item in args.cot_arg:
            key, value = item.split("=", 1)
            for o in cfg["objectives"]:
                if o["name"] == "cot_term":
                    o.setdefault("args", {})[key] = yaml.safe_load(value)
        for item in args.weight:
            name, value = item.split("=", 1)
            hit = [o for o in cfg["objectives"] if o["name"] == name]
            if not hit:
                raise SystemExit(f"--weight: no objective {name} in the config")
            hit[0]["weight"] = float(value)
        if args.linear_solver:
            cfg["settings"]["linear_solver"] = args.linear_solver
        if args.max_iter:
            cfg["settings"]["max_iter"] = args.max_iter
        if args.log:
            cfg["settings"].update({"enable_logging": True, "iteration_log_interval": 25, "launch_dashboard": args.dashboard})
        lo, hi = duration_bounds(speed if args.free_speed is None else args.free_speed[0])
        cfg["bounds"]["dur"] = [args.dur_min or lo, args.dur_max or hi]

    config = pc.write_config(pc.WALKING_TEMPLATE, speed_dir(out_dir, speed, args.variant) / f"_config{args.tag}.yaml", edit)
    problem = pc.ReusableProblem(config)
    log(f"walking problem built (setup {problem.setup_seconds:.1f} s)")

    summary = speed_dir(out_dir, speed, args.variant) / f"summary{args.tag}.csv"
    columns = ["seed", "standing_status", "standing_seconds", "walking_status", "converged", "objective", "terms",
               "duration_half_cycle", "speed", "walking_seconds"]
    new_file = not summary.exists()
    handle = open(summary, "a", newline="")
    writer = csv.writer(handle)
    if new_file:
        writer.writerow(columns)

    converged = 0
    attempts = 0
    for seed in (args.seeds or range(1, args.max_attempts + 1)):
        if converged >= args.n_converged:
            break
        attempts += 1
        path = walking_path(out_dir, speed, seed, args.variant)
        if path.exists():
            (x, g), info, _ = cloudpickle.load(open(path, "rb"))
            done = int(info["status"]) == 0
            converged += done
            log(f"seed {seed}: result exists (status {int(info['status'])}), converged so far {converged}")
            continue

        if warm:
            nominal = guess_file(seed)
            if not nominal.exists() or int(cloudpickle.load(open(nominal, "rb"))[1]["status"]) != 0:
                log(f"seed {seed}: no converged nominal solution in {warm}, skipped")
                continue
            standing_status, standing_seconds = None, None
        else:
            standing.run(seed)
            standing_status, standing_seconds = standing.stored(seed)
        result = problem.solve(problem.x0_from_file(guess_file(seed)), out=path)
        converged += result["converged"]
        duration = float(np.ravel(result["x"][1].dur)[0])
        achieved = float(np.ravel(result["x"][1].speed)[0])
        writer.writerow([seed, standing_status, standing_seconds, result["status"], int(result["converged"]),
                         result["objective"], result["terms"], duration, f"{achieved:.4f}",
                         f"{result['solve_seconds']:.1f}"])
        handle.flush()
        log(f"seed {seed}: walking status {result['status']}, objective {result['objective']:.4f}, "
            f"terms {result['terms']}, dur {duration:.4f} s, {result['solve_seconds']:.0f} s; converged {converged}/{args.n_converged}")

    handle.close()
    log(f"done: {converged} converged in {attempts} attempts, summary {summary}")
    return 0 if converged >= args.n_converged else 1


if __name__ == "__main__":
    sys.exit(main())
