"""
Measure wall-clock solve time for the completed IMU-tracking trials and write
result/runtime_measurements.json, consumed by scripts/report.py.

Why this exists: the original result/*.pkl files don't contain timing info
(the solver never recorded it), and the current biosym working tree usually
has in-progress, uncommitted changes that are incompatible with configs/ig
files from whenever a given result was actually produced (state-vector
dimensions, contact-model state counts, etc. drift as biosym evolves). So
this script:

  1. Creates an isolated git worktree of biosym at HEAD (a clean, committed
     revision -- not necessarily bit-identical to whatever produced the
     existing results, but a much better bet than the live working tree).
  2. Patches that worktree's collocation.py to record wall_time_setup/
     wall_time_solve/wall_time_total in the IPOPT info dict, and to not
     auto-launch the interactive Dash dashboard (which hangs/crashes in a
     non-interactive batch run).
  3. Backs up each trial's result/*.pkl, re-solves it against the worktree,
     records the timing, then restores the original .pkl -- the re-solve is
     only trusted for timing, not for the solution itself (it can converge to
     a different objective value than the original saved result).
  4. Trials whose warm-start (ig_*.pkl) files are incompatible with the HEAD
     worktree are skipped with a warning, not silently dropped.

Usage:
    python scripts/measure_runtime.py [--worktree PATH] [--keep-worktree]
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)

# scripts/report.py's discover_trials(), duplicated to keep this script
# runnable standalone (`python scripts/measure_runtime.py`) without relying
# on report.py's module-level side effects (it chdirs and does I/O at import
# time via TOTAL_PARTICIPANTS/TOTAL_CONDITIONS).
def discover_trials():
    pairs = []
    for path in sorted(glob.glob("result/P*/*.pkl")):
        subject = os.path.basename(os.path.dirname(path))
        trial = os.path.splitext(os.path.basename(path))[0]
        if trial.startswith("ig_"):
            continue
        pairs.append((subject, trial))
    return pairs


def find_biosym_repo():
    """Locate the biosym repo this project depends on, via `import biosym`."""
    import biosym

    return os.path.dirname(os.path.dirname(os.path.abspath(biosym.__file__)))


def patch_collocation_timing(collocation_py):
    """Add wall_time_setup/solve/total recording, and disable the interactive
    dashboard (which isn't safe in a non-interactive batch re-solve)."""
    src = open(collocation_py).read()
    if "wall_time_total" in src:
        return  # already patched
    assert "import os\n" in src
    src = src.replace("import os\n", "import os\nimport time\n", 1)
    src = src.replace(
        "        # Create a cyipopt.problem\n",
        "        _setup_start = time.perf_counter()\n        # Create a cyipopt.problem\n",
        1,
    )
    src = src.replace(
        '        self.iteration_logger = None\n',
        '        self.iteration_logger = None\n        self.wall_time_setup = time.perf_counter() - _setup_start\n',
        1,
    )
    src = src.replace(
        "        x, info = self.nlp.solve(self.x0)\n",
        "        _solve_start = time.perf_counter()\n"
        "        x, info = self.nlp.solve(self.x0)\n"
        '        info["wall_time_solve"] = time.perf_counter() - _solve_start\n'
        '        info["wall_time_setup"] = self.wall_time_setup\n'
        '        info["wall_time_total"] = info["wall_time_solve"] + self.wall_time_setup\n',
        1,
    )
    src = src.replace(
        'launch_dashboard = self.settings.get("settings", {}).get("launch_dashboard", True)',
        "launch_dashboard = False  # disabled for offline/batch timing reruns in this worktree",
        1,
    )
    with open(collocation_py, "w") as f:
        f.write(src)


def measure_trial(subject, trial, worktree_biosym):
    result_path = f"result/{subject}/{trial}.pkl"
    backup_path = result_path + ".runtime_backup"
    shutil.copy(result_path, backup_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = worktree_biosym + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/run_experiment.py", subject, trial],
            cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            print(f"  [skip] {subject}/{trial}: re-solve failed (see below)")
            print("\n".join(proc.stdout.splitlines()[-5:] + proc.stderr.splitlines()[-15:]))
            return None
        with open(result_path, "rb") as f:
            (_states, _globals), info, _settings = pickle.load(f)
        if "wall_time_total" not in info:
            print(f"  [skip] {subject}/{trial}: solved but no wall_time_total in info (patch didn't apply?)")
            return None
        return {k: info[k] for k in ("wall_time_setup", "wall_time_solve", "wall_time_total")}
    finally:
        shutil.move(backup_path, result_path)  # always restore the original result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worktree", default=os.path.join(os.path.dirname(PROJECT_DIR), "..", "biosym_runtime_worktree"),
                         help="Where to create the isolated biosym worktree")
    parser.add_argument("--keep-worktree", action="store_true", help="Don't remove the worktree afterward")
    args = parser.parse_args()
    worktree = os.path.abspath(args.worktree)

    biosym_repo = find_biosym_repo()
    trials = discover_trials()
    if not trials:
        raise SystemExit("No completed (non-ig_*) trials found under result/.")

    print(f"Creating isolated worktree at {worktree} (biosym HEAD)...")
    subprocess.run(["git", "worktree", "add", worktree, "HEAD"], cwd=biosym_repo, check=True)
    try:
        patch_collocation_timing(os.path.join(worktree, "biosym", "ocp", "collocation.py"))

        measurements = {}
        for subject, trial in trials:
            print(f"Measuring {subject}/{trial}...")
            timing = measure_trial(subject, trial, worktree)
            if timing is not None:
                measurements[f"{subject}/{trial}"] = timing
                print(f"  wall_time_total = {timing['wall_time_total']:.1f} s")

        out_path = "result/runtime_measurements.json"
        with open(out_path, "w") as f:
            json.dump(measurements, f, indent=2, sort_keys=True)
        print(f"\nWrote {len(measurements)}/{len(trials)} measurements to {out_path}")
    finally:
        if not args.keep_worktree:
            subprocess.run(["git", "worktree", "remove", worktree, "--force"], cwd=biosym_repo, check=False)


if __name__ == "__main__":
    main()
