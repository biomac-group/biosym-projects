"""
Standing of the BiomechPriorVAE paper pipeline, for one or several seeds.

    python scripts/paper_standing.py SEED [SEED ...] [--linear-solver mumps] [--out-dir results] [--force]

Per seed (every random draw comes from ``numpy.random.default_rng(seed)``):

1. sample a random standing pose and random foot forces, with zero joint velocities;
2. optimise the joint angles and the foot forces on the learned state prior (the BiomechPriorVAE reconstruction NLL);
3. find the pelvis height at which the contact law carries one body weight vertically;
4. equilibrate the muscles at that pose (fibre lengths at force equilibrium, activation 0.02);
5. solve the standing problem (``configs/standing_paper.yaml``: muscle and arm effort, static equilibrium) from this guess.

The result of a seed is ``results/standing/seed<SEED>.pkl``, in the format of ``Collocation.solve``; the walking
script (``paper_walking.py``) starts from it. The standing problem is built once per process and every seed solves it
with ``nlp.solve(x0)``.
"""
import argparse
import sys
import time
from pathlib import Path

import cloudpickle
import jax.numpy as jnp
import numpy as np

import paper_common as pc


class Standing:
    """
    The standing pipeline of the paper with its problem built once.

    Parameters
    ----------
    out_dir : str or Path, optional
        Folder of the results.
    linear_solver : str, optional
        IPOPT linear solver (``"ma97"``, ``"mumps"``, ...); the default is biosym's.
    """

    def __init__(self, out_dir=pc.RESULT_DIR, linear_solver=None):
        self.out_dir = Path(out_dir)
        stopwatch = pc.Stopwatch()

        def edit(cfg):
            if linear_solver:
                cfg["settings"]["linear_solver"] = linear_solver

        config = pc.write_config(pc.STANDING_TEMPLATE, self.out_dir / "standing" / "_standing.yaml", edit)
        self.problem = pc.ReusableProblem(config)
        self.model = self.problem.model
        bounds = self.problem.settings["bounds"]
        self.q_lo, self.q_hi = np.asarray(bounds["min"].q)[0], np.asarray(bounds["max"].q)[0]
        self.act_lo, self.act_hi = np.asarray(bounds["min"].actuator_model)[0], np.asarray(bounds["max"].actuator_model)[0]
        self.height_index = [str(n).removeprefix("q_") for n in self.model.coordinates.names].index("pelvis_ty")
        self.vertical = pc.vertical_force_function(self.model)
        self.setup_seconds = self.problem.setup_seconds
        stopwatch(f"standing problem built (setup {self.setup_seconds:.1f} s)")

    def path(self, seed):
        """Result file of a seed."""
        return self.out_dir / "standing" / f"seed{seed}.pkl"

    def stored(self, seed):
        """Status and solve time of the stored result of a seed."""
        _, info, _ = cloudpickle.load(open(self.path(seed), "rb"))
        return int(info["status"]), float(info.get("wall_time_solve", float("nan")))

    def run(self, seed, force=False, verbose=True):
        """
        Run the standing pipeline for one seed.

        Parameters
        ----------
        seed : int
            Seed of the random pose.
        force : bool, optional
            Redo a seed whose result file exists.
        verbose : bool, optional
            Print the stages.

        Returns
        -------
        dict
            ``path``, ``status``, ``objective``, ``solve_seconds``, ``guess_seconds``, ``height``, ``prior_nll`` (start,
            optimised), ``muscle_residual``; ``status`` is ``None`` for a seed that was already done.
        """
        path = self.path(seed)
        if path.exists() and not force:
            return {"path": path, "status": None}
        log = pc.Stopwatch() if verbose else (lambda message: None)
        t_start = time.perf_counter()
        rng = np.random.default_rng(seed)

        pose = pc.optimise_pose_on_prior(self.model, self.q_lo, self.q_hi, rng)
        log(f"seed {seed}: prior pose, NLL {pose['nll_start']:.3f} -> {pose['nll_end']:.3f}, "
            f"foot forces [Fy_r, Flat_r, Fy_l, Flat_l] = {np.round(pose['force'], 3)}")
        q = pose["q"]
        q[self.height_index], force = pc.solve_pelvis_height(self.vertical, q, self.height_index, target=1.0)
        height = float(q[self.height_index])
        q = np.clip(q, self.q_lo, self.q_hi)
        log(f"seed {seed}: pelvis_ty {height:.4f} m carries {force:.4f} body weights")

        block, residual, lce = pc.equilibrate_muscles(self.model, q)
        block = np.clip(block, self.act_lo, self.act_hi)
        log(f"seed {seed}: muscles equilibrated, Lce in [{lce.min():.3f}, {lce.max():.3f}], max |residual| {residual:.1e} F_max")

        guess = self.problem.ocp.initial_guess_states
        guess = guess.replace(q=jnp.asarray(q)[None], qd=jnp.zeros_like(guess.qd),
                              actuator_model=jnp.asarray(block)[None], gc_model=jnp.zeros_like(guess.gc_model))
        guess = pc.initial_guess_utils._with_contact_states(guess, self.model)      # the contact law's forces
        guess_seconds = time.perf_counter() - t_start

        result = self.problem.solve(self.problem.x0(guess), out=path)
        log(f"seed {seed}: standing solved, status {result['status']}, objective {result['objective']:.6g}, "
            f"{result['solve_seconds']:.1f} s")
        return {"path": path, "status": result["status"], "objective": result["objective"], "terms": result["terms"],
                "solve_seconds": result["solve_seconds"], "guess_seconds": guess_seconds, "height": height,
                "prior_nll": (pose["nll_start"], pose["nll_end"]), "muscle_residual": residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seeds", type=int, nargs="+", help="seeds of the random poses")
    parser.add_argument("--linear-solver", default=None, help="IPOPT linear solver, e.g. ma97 or mumps")
    parser.add_argument("--out-dir", default=str(pc.RESULT_DIR))
    parser.add_argument("--force", action="store_true", help="redo seeds whose result exists")
    args = parser.parse_args()

    standing = Standing(args.out_dir, args.linear_solver)
    rows = [standing.run(seed, force=args.force) | {"seed": seed} for seed in args.seeds]
    print("\nseed  status  objective      solve [s]  guess [s]  pelvis_ty [m]  prior NLL (start -> optimised)")
    for r in rows:
        if r["status"] is None:
            print(f"{r['seed']:4d}  done before ({r['path']})")
        else:
            print(f"{r['seed']:4d}  {r['status']:6d}  {r['objective']:12.6g}  {r['solve_seconds']:9.1f}  "
                  f"{r['guess_seconds']:9.1f}  {r['height']:13.4f}  {r['prior_nll'][0]:.3f} -> {r['prior_nll'][1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
