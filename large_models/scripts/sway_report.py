"""
Lateral sway of a walking solution and where the torques on its coordinates come from.

    python scripts/sway_report.py MODEL RESULT.pkl

Prints the range of the lateral coordinates (pelvis_tz, pelvis_list, the spine's lateral bending, hip adduction) and, for
the spinal and hip lateral coordinates, the torque split over the half cycle (peak-to-peak and RMS in N m): the net torque
the motion requires (inverse dynamics, qdd by backward differences), the active and the passive muscle torque, the
passive joint torque (spring, damper, range walls) and the remainder, which should be ~0.
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
from biosym.model.model import BiosymModel  # noqa: E402
from biosym.utils.aba import get_rnea_biosym  # noqa: E402

name, result = sys.argv[1], sys.argv[2]
m = BiosymModel(os.path.join(PROJECT, "models", name, f"{name}.yaml"), compile_eom=False, fit_muscle_paths=False)
names = [n.removeprefix("q_") for n in m.coordinates.names]
(x, g), info, s = cloudpickle.load(open(result, "rb"))
n = int(s["nnodes"])
q, qd = np.asarray(x.q[:n]), np.asarray(x.qd[:n])
h = float(g.dur) / (int(s["nnodes_dur"]) - 1)
qdd = np.vstack([np.zeros_like(qd[:1]), np.diff(qd, axis=0) / h])
deg = lambda c: np.degrees(q[:, names.index(c)])
lat = [c for c in names if c in ("pelvis_tz", "pelvis_list", "L5_S1_LB", "T12_L1_LB", "T1_head_neck_LB", "lat_bending", "hip_adduction_r", "hip_adduction_l")]
print(f"{name}: lateral coordinates over the half cycle (min .. max):")
for c in lat:
    v = q[:, names.index(c)]
    print(f"  {c:16s} {(1e3 * v.min() if c == 'pelvis_tz' else np.degrees(v.min())):8.1f} .. {(1e3 * v.max() if c == 'pelvis_tz' else np.degrees(v.max())):8.1f} {'mm' if c == 'pelvis_tz' else 'deg'}")
mus, c = m.actuator_model, m.default_constants
rnea = get_rnea_biosym(m)
rows = {k: [] for k in ("id", "act", "pas", "joint")}
for i in range(1, n):
    st = m.default_states.replace(q=jnp.asarray(q[i]), qd=jnp.asarray(qd[i]), qdd=jnp.asarray(qdd[i]),
                                  actuator_model=jnp.asarray(x.actuator_model[i]), gc_model=jnp.asarray(x.gc_model[i]))
    act, pas, ma = (np.asarray(v) for v in mus._forces(st))
    ma = ma[:, :, 0]
    rows["act"].append(ma.T @ act[:, 0]); rows["pas"].append(ma.T @ pas[:, 0])
    rows["joint"].append(np.asarray(m.passive_actuators.forward(st, c, m)).reshape(-1))
    rows["id"].append(np.asarray(rnea(st, c)).reshape(-1))
R = {k: np.array(v) for k, v in rows.items()}
print("torque split on the lateral coordinates [N m]: peak-to-peak (RMS):")
print(f"  {'coordinate':16s} {'required':>16s} {'active muscle':>16s} {'passive muscle':>16s} {'passive joint':>16s} {'remainder':>12s}")
for cname in [c for c in lat if c != "pelvis_tz" and c != "pelvis_list"]:
    j = names.index(cname)
    f = lambda a: f"{np.ptp(a[:, j]):6.1f} ({np.sqrt(np.mean(a[:, j] ** 2)):5.1f})"
    rem = R["id"][:, j] - (R["act"][:, j] + R["pas"][:, j] + R["joint"][:, j])
    rem2 = R["id"][:, j] + (R["act"][:, j] + R["pas"][:, j] + R["joint"][:, j])
    print(f"  {cname:16s} {f(R['id']):>16s} {f(R['act']):>16s} {f(R['pas']):>16s} {f(R['joint']):>16s} {min(np.abs(rem).max(), np.abs(rem2).max()):12.2f}")
