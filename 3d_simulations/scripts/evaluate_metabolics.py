"""
Post-evaluate the metabolic cost of converged results with the Umberger and the Bhargava energy models.

    python scripts/evaluate_metabolics.py FOLDER [FOLDER ...]      # e.g. results/walking/v1.20 results/effort/v3.50

For every converged ``seed<k>.pkl`` of a folder the cost of transport ``(mean muscle energy rate / mass + 1 W/kg) /
speed`` [J/kg/m] is computed with ``cot_term`` (``metabolic_model: umberger`` and ``bhargavaact``, the toolbox's models,
forward-difference fibre velocity) and written to ``<folder>/metabolics.csv`` with the columns ``seed``, ``speed``,
``umberger`` and ``bhargava``. The problem is built from the folder's ``_config*.yaml``.
"""
import argparse
import csv
import glob
import re
from pathlib import Path

import cloudpickle
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from biosym.objectives import cot_term                                           # noqa: E402
from biosym.ocp import collocation                                              # noqa: E402


def evaluate(folder):
    """Write and return the metabolic costs of the converged results of one folder."""
    folder = Path(folder)
    ocp = collocation.Collocation(str(sorted(folder.glob("_config*.yaml"))[0]))
    settings = ocp.objective.settings
    costs = {name: cot_term.Objective(ocp.model, settings, metabolic_model=name, exponent=1) for name in
             ("umberger", "bhargavaact")}
    funs = {name: obj.get_objfun() for name, obj in costs.items()}
    gravity = costs["umberger"]._get_info()["gravity"]
    rows = []
    for path in sorted(glob.glob(str(folder / "seed*.pkl"))):
        (states, globals_), result, _ = cloudpickle.load(open(path, "rb"))
        if int(result["status"]) != 0:
            continue
        values = {name: float(fun(states, globals_)) * gravity for name, fun in funs.items()}
        rows.append((int(re.search(r"seed(\d+)", path).group(1)), float(np.linalg.norm(np.ravel(globals_.speed))),
                     values["umberger"], values["bhargavaact"]))
    out = folder / "metabolics.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "speed", "umberger", "bhargava"])
        writer.writerows(rows)
    print(f"{out}: {len(rows)} converged, Umberger {np.mean([r[2] for r in rows]):.2f}, "
          f"Bhargava {np.mean([r[3] for r in rows]):.2f} J/kg/m (mean)")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    for folder in parser.parse_args().folders:
        evaluate(folder)


if __name__ == "__main__":
    main()
