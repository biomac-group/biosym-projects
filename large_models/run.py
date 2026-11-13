"""
Walk at 1.3 m/s with a model of this project and the learned state prior.

    python run.py MODEL [effort|metabolic]        # MODEL: ms_human_700 or myofullbody; default objective: effort

``effort`` (weight 300, cubed, volume weighted) is the pipeline of the 3D example: a random standing pose on the prior
(``make_prior_guess``, seed 1, through the coordinate map of ``objectives/prior_map.py``), solved for quiet standing, then
the walking half cycle (``configs/MODEL_standing.yaml``, ``configs/MODEL_walking.yaml``).

``metabolic`` (log cost of transport of the smooth Bhargava model, ``configs/MODEL_walking_metabolic.yaml``) starts from the
effort solution of the walking half cycle, which is made first when it does not exist.

Results are in ``results/MODEL/``; the viewer stays open at the end. The model files are made by the scripts in
``scripts/`` (see README.md).
"""
import os
import sys

from biosym.ocp.collocation import Collocation
from biosym.ocp.utils.prior_guess import make_prior_guess
from biosym.ocp.utils.vectorize import states_dict_to_x

from objectives.prior_map import PriorCoordinates

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from model_config import from_argv  # noqa: E402

cfg = from_argv()
objective = sys.argv[2] if len(sys.argv) > 2 else "effort"
if objective not in ("effort", "metabolic"):
    raise SystemExit("usage: run.py MODEL [effort|metabolic]")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # the yaml paths are relative to this folder

if not os.path.exists(f"results/{cfg.name}/standing_seed1.pkl"):
    standing = Collocation(f"configs/{cfg.name}_standing.yaml")
    guess = make_prior_guess(standing.model, standing.settings, standing.initial_guess_states, seed=1,
                             coordinate_map=PriorCoordinates(standing.model, cfg.thorax))
    standing.x0 = states_dict_to_x(guess["states"], None)
    standing.solve()

if objective == "metabolic" and not os.path.exists(f"results/{cfg.name}/walking_v1.30_seed1.pkl"):
    Collocation(f"configs/{cfg.name}_walking.yaml").solve()          # the start of the metabolic problem

walking = Collocation(f"configs/{cfg.name}_walking{'_metabolic' if objective == 'metabolic' else ''}.yaml")
walking.solve()
walking.view(port=cfg.viewer_port)
