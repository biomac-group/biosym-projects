"""
Solve one collocation config of this project and print the objective terms at the solution.

    python scripts/solve.py configs/myofullbody_walking_warm.yaml

The paths in the config are relative to the project folder, which this script changes into.
"""
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from biosym.ocp.collocation import Collocation  # noqa: E402

ocp = Collocation(sys.argv[1])
solution = ocp.solve()
print("SOLVED", sys.argv[1], "status", solution.info.get("status") if isinstance(solution.info, dict) else solution.info)
for term, value in solution.objective_values.items():
    print(f"  objective {term} (weighted): {value:.4f}")
