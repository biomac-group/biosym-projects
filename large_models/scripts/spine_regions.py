"""
Bending of the spine by region for the six walking runs: the lumbar spine and the rest (neck and head).

    python scripts/spine_regions.py

Each region is the orientation of a segment relative to its neighbour, measured from the default pose and decomposed in
the gait3d lumbar sequence (z extension, x lateral bending, y axial rotation; `objectives.prior_map.zxy_angles`), so it
sums every joint in between:

* lumbar: the thorax relative to the pelvis (SIPP: ``torso``, one joint; MS-Human: ``thoracic12``, L5_S1 and T12_L1 in
  series; MyoFullBody: ``lumbar1``, five coupled levels);
* rest of the spine: the head relative to the thorax, only where the model has neck joints (MS-Human: ``head_neck``). SIPP's
  and MyoFullBody's head is rigid with the thorax.
"""
import os
import sys

import cloudpickle
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from biosym.model.contact.body_kinematics import BodyKinematics  # noqa: E402
from biosym.model.model import BiosymModel  # noqa: E402
from objectives.prior_map import zxy_angles  # noqa: E402
import render_compare as rc  # noqa: E402

PROJECTS = os.path.dirname(PROJECT)
LARGE = os.path.join(PROJECT, "results")
MODELS = {
    "SIPP": (os.path.join(PROJECTS, "3d_simulations/models/sipp_runmad/sipp_runmad.yaml"), "torso", None),
    "MS-Human": (os.path.join(PROJECT, "models/ms_human_700/ms_human_700.yaml"), "thoracic12", "head_neck"),
    "MyoFullBody": (os.path.join(PROJECT, "models/myofullbody/myofullbody.yaml"), "lumbar1", None),
}
RUNS = [
    ("SIPP", "effort", str(rc.best_sipp("effort", 1.3)[0])), ("SIPP", "metabolic", str(rc.best_sipp("walking", 1.3)[0])),
    ("MS-Human", "effort", f"{LARGE}/ms_human_700/walking_v1.30_seed1.pkl"),
    ("MS-Human", "metabolic", f"{LARGE}/ms_human_700/walking_metabolic_v1.30_seed1.pkl"),
    ("MyoFullBody", "effort", f"{LARGE}/myofullbody/walking_v1.30_seed1.pkl"),
    ("MyoFullBody", "metabolic", f"{LARGE}/myofullbody/walking_metabolic_v1.30_seed1.pkl"),
]


def region_angles(kin, consts, q, parent, child, rest):
    """Angles [deg] (extension, bending, rotation) of ``child`` relative to ``parent`` for the coordinates ``q`` (nodes, n_q)."""
    def one(qq):
        R, _ = kin.poses(qq, consts)
        A = lambda b: R[kin.index(b)] @ jnp.asarray(rest[b]).T
        return jnp.stack(zxy_angles(A(parent).T @ A(child)))
    return np.degrees(np.asarray(jax.vmap(one)(jnp.asarray(q))))


def main():
    cache = {}
    print("angles of the segment relative to its neighbour over the half cycle [deg]: min .. max (range)")
    for label, objective, pkl in RUNS:
        yaml, thorax, head = MODELS[label]
        if label not in cache:
            m = BiosymModel(yaml, compile_eom=False, fit_muscle_paths=False)
            kin = BodyKinematics(m)
            consts = jnp.asarray(m.default_constants.filter("model").to_array())
            R0, _ = kin.poses(jnp.zeros(m.coordinates.n), consts)
            rest = {b: np.asarray(R0[kin.index(b)]) for b in ("pelvis", thorax) + ((head,) if head else ())}
            cache[label] = (kin, consts, rest)
        kin, consts, rest = cache[label]
        (x, g), info, s = cloudpickle.load(open(pkl, "rb"))
        q = np.asarray(x.q[:int(s["nnodes"])])
        regions = [("lumbar", "pelvis", thorax)] + ([("neck/head", thorax, head)] if head else [])
        parts = []
        for name, parent, child in regions:
            a = region_angles(kin, consts, q, parent, child, rest)
            parts.append(f"{name}: bending {a[:, 1].min():5.1f}..{a[:, 1].max():5.1f} ({np.ptp(a[:, 1]):4.1f}), "
                         f"extension {a[:, 0].min():6.1f}..{a[:, 0].max():6.1f} ({np.ptp(a[:, 0]):4.1f}), "
                         f"rotation {a[:, 2].min():5.1f}..{a[:, 2].max():5.1f} ({np.ptp(a[:, 2]):4.1f})")
        if not head:
            parts.append("neck/head: rigid with the thorax")
        print(f"{label:11s} {objective:9s} " + " || ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
