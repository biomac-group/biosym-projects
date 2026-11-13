"""
Does the spine above the lowest lumbar joint change the motion of the upper body?

    python scripts/spine_segments.py

SIPP has one spinal joint (pelvis to torso). MS-Human and MyoFullBody have more: the lowest lumbar joint (pelvis to
``lumbar5``, the counterpart of SIPP's joint) and the levels above it up to the thorax (MS-Human ``T12_L1``, MyoFullBody
``lumbar4`` .. ``lumbar1``). For every run the script prints, as ranges over the half cycle (degrees; bending = lateral lean,
rotation = about the vertical; orientations from the default pose, `objectives.prior_map.zxy_angles`):

* the angle of the lower joint and of the upper levels (the thorax relative to ``lumbar5``);
* the world orientation of the pelvis, of ``lumbar5`` and of the thorax. ``lumbar5`` is what the thorax would do if the
  upper levels were locked (SIPP's situation); the thorax's actual range against it shows what the upper levels add or
  take away;
* the angular velocity of the thorax in the world (RMS, deg/s), a measure of how restless the upper body is.
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
from spine_regions import MODELS, RUNS  # noqa: E402

LOWER = {"SIPP": None, "MS-Human": "lumbar5", "MyoFullBody": "lumbar5"}      # body above the lowest lumbar joint


def main():
    cache = {}
    for label, objective, pkl in RUNS:
        yaml, thorax, head = MODELS[label]
        if label not in cache:
            m = BiosymModel(yaml, compile_eom=False, fit_muscle_paths=False)
            kin = BodyKinematics(m)
            consts = jnp.asarray(m.default_constants.filter("model").to_array())
            R0, _ = kin.poses(jnp.zeros(m.coordinates.n), consts)
            bodies = ["pelvis", thorax] + ([LOWER[label]] if LOWER[label] else [])
            cache[label] = (m, kin, consts, {b: np.asarray(R0[kin.index(b)]) for b in bodies}, bodies)
        m, kin, consts, rest, bodies = cache[label]
        (x, g), info, s = cloudpickle.load(open(pkl, "rb"))
        n = int(s["nnodes"])
        q = np.asarray(x.q[:n])
        h = float(np.ravel(g.dur)[0]) / (int(s["nnodes_dur"]) - 1)

        def world(qq):
            R, _ = kin.poses(qq, consts)
            return {b: R[kin.index(b)] @ jnp.asarray(rest[b]).T for b in bodies}

        W = jax.vmap(world)(jnp.asarray(q))                      # body -> (n, 3, 3) rotation away from the default pose
        ang = lambda R: np.degrees(np.asarray(jax.vmap(lambda r: jnp.stack(zxy_angles(r)))(R)))   # (n, 3): ext, bend, rot
        rel = lambda a, b: ang(jnp.einsum("nji,njk->nik", W[a], W[b]))
        out = [f"{label:11s} {objective:9s}"]
        rng = lambda a, c: f"{np.ptp(a[:, c]):5.1f}"
        if LOWER[label]:
            lower, upper = rel("pelvis", LOWER[label]), rel(LOWER[label], thorax)
            out.append(f"lower joint bend {rng(lower, 1)} rot {rng(lower, 2)} | upper levels bend {rng(upper, 1)} rot {rng(upper, 2)}")
            wl, wt = ang(W[LOWER[label]]), ang(W[thorax])
            out.append(f"world: pelvis bend {rng(ang(W['pelvis']), 1)} rot {rng(ang(W['pelvis']), 2)} | lumbar5 (upper locked) bend {rng(wl, 1)} rot {rng(wl, 2)} | thorax bend {rng(wt, 1)} rot {rng(wt, 2)}")
        else:
            lower = rel("pelvis", thorax)
            wt = ang(W[thorax])
            out.append(f"single joint bend {rng(lower, 1)} rot {rng(lower, 2)}")
            out.append(f"world: pelvis bend {rng(ang(W['pelvis']), 1)} rot {rng(ang(W['pelvis']), 2)} | thorax bend {rng(wt, 1)} rot {rng(wt, 2)}")
        # angular velocity of the thorax in the world: |R_dot R^T| by central differences on the periodic half cycle (mirror)
        Rt = np.asarray(W[thorax])
        d = (Rt[2:] - Rt[:-2]) / (2 * h)
        omega = np.array([np.linalg.norm([(d[i] @ Rt[i + 1].T)[2, 1], (d[i] @ Rt[i + 1].T)[0, 2], (d[i] @ Rt[i + 1].T)[1, 0]]) for i in range(len(d))])
        out.append(f"thorax angular velocity RMS {np.degrees(np.sqrt(np.mean(omega ** 2))):5.1f} deg/s")
        print(" | ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
