"""
Break the state prior's cost of a walking solution down by feature, and list the arm and wrist angles.

    python scripts/prior_report.py MODEL RESULT.pkl

Prints the mean prior NLL per node (the value of ``state_prior_term`` with weight 1), the features with the highest mean
NLL with their value against the prior's training mean and spread, and the arm and wrist angles over the half cycle.
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
from biosym.constraints.dynamics import calc_forces  # noqa: E402
from biosym.model.model import BiosymModel  # noqa: E402
from biosym.objectives.state_prior_term import Objective as PriorTerm, _foot_force  # noqa: E402
from biosym.utils import state_prior  # noqa: E402
from model_config import MODELS  # noqa: E402
from objectives.prior_map import PriorCoordinates  # noqa: E402


def main():
    cfg = MODELS[sys.argv[1]]
    m = BiosymModel(os.path.join(PROJECT, "models", cfg.name, f"{cfg.name}.yaml"), compile_eom=False, fit_muscle_paths=False)
    names = [n.removeprefix("q_") for n in m.coordinates.names]
    (x, g), info, settings = cloudpickle.load(open(sys.argv[2], "rb"))
    n = int(settings["nnodes"])
    prior = state_prior.load_state_prior()
    pn = prior["coord_names"]
    cmap = PriorCoordinates(m, cfg.thorax)
    qp, qdp = jax.vmap(lambda a, b: jax.jvp(cmap, (a,), (b,)))(jnp.asarray(x.q[:n]), jnp.asarray(x.qd[:n]))
    filled = jax.vmap(calc_forces, in_axes=(0, None, None))(x[:n], m, m.default_constants)
    idx = {side: PriorTerm._side_indices(m.ext_forces.names, tag) for side, tag in (("right", "r"), ("left", "l"))}
    bw = float(m.reference_body_weight())
    feats = state_prior.features_from_state(qp, qdp, _foot_force(filled.ext_forces, idx["right"], bw),
                                            _foot_force(filled.ext_forces, idx["left"], bw),
                                            [pn.index(k) for k in prior["knee_names"]])
    P, meta = prior["params"], prior["meta"]
    mean, scale = jnp.asarray(P["scaler_mean"]), jnp.asarray(P["scaler_scale"])
    mu, var = state_prior.prior_forward((feats - mean) / scale, P, meta)
    recon, var = mu * scale + mean, jnp.maximum(var * scale ** 2, float(meta["nll_var_floor"]))
    nll = np.asarray(0.5 * (jnp.log(var) + (recon - feats) ** 2 / var))
    labels = list(pn) + [f"d_{c}" for c in pn] + ["Fy_r", "Flat_r", "Fy_l", "Flat_l"]
    print(f"prior NLL per node: mean {nll.mean():.3f} (coordinates {nll[:, :23].mean():.3f}, velocities "
          f"{nll[:, 23:46].mean():.3f}, forces {nll[:, 46:].mean():.3f})")
    excess = nll - np.asarray(0.5 * jnp.log(var))          # the misfit part, (recon - x)^2 / (2 var)
    print("features with the largest misfit (recon - value)^2 / 2 var, mean over nodes; coordinates in deg:")
    for i in np.argsort(-excess.mean(0))[:10]:
        conv = (lambda v: np.degrees(v)) if i < 46 else (lambda v: v)
        print(f"  {labels[i]:18s} misfit {excess[:, i].mean():6.2f} | value {float(conv(np.asarray(feats)[:, i].mean())):8.2f}"
              f", prior reconstruction {float(conv(np.asarray(recon)[:, i].mean())):8.2f}, training {float(conv(mean[i])):7.2f}"
              f" +- {float(conv(scale[i])):6.2f}")
    print("arm and wrist angles over the half cycle [deg] (min / mean / max):")
    for c in ("shoulder_elv_r", "shoulder_rot_r", "elbow_flexion_r", "pro_sup_r", "flexion_r", "deviation_r",
              "shoulder_elv_l", "elbow_flexion_l", "pro_sup_l", "flexion_l", "deviation_l"):
        v = np.degrees(np.asarray(x.q[:n, names.index(c)]))
        print(f"  {c:16s} {v.min():7.1f} {v.mean():7.1f} {v.max():7.1f}")
    print("mapped prior arm coordinates [deg] (mean): " + ", ".join(
        f"{c} {np.degrees(np.asarray(qp[:, pn.index(c)]).mean()):.1f}" for c in pn[13:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
