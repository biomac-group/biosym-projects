"""
Bound contact and muscle effort of a walking solution.

    python scripts/bounds_and_effort.py MODEL RESULT.pkl

Lists the coordinates (and velocities) within 5 % of a bound, and the muscles and regions that carry the cubed,
volume-weighted muscle effort (the term of the objective): share of the total by muscle group.
"""
import os
import re
import sys
from collections import defaultdict

import cloudpickle
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from biosym.model.model import BiosymModel  # noqa: E402

name, result = sys.argv[1], sys.argv[2]
m = BiosymModel(os.path.join(PROJECT, "models", name, f"{name}.yaml"), compile_eom=False, fit_muscle_paths=False)
names = [n.removeprefix("q_") for n in m.coordinates.names]
(x, g), info, s = cloudpickle.load(open(result, "rb"))
n = int(s["nnodes"])
b = s["bounds"]
lo, hi = np.asarray(b["min"].q).reshape(-1, len(names))[0], np.asarray(b["max"].q).reshape(-1, len(names))[0]
vlo, vhi = np.asarray(b["min"].qd).reshape(-1, len(names))[0], np.asarray(b["max"].qd).reshape(-1, len(names))[0]
q, qd = np.asarray(x.q[:n]), np.asarray(x.qd[:n])
print(f"{name}: coordinates within 5 % of their range from a bound (position, deg):")
for i, c in enumerate(names):
    if c.startswith("pelvis_t"):
        continue
    d = min(q[:, i].min() - lo[i], hi[i] - q[:, i].max()) / (hi[i] - lo[i])
    if d < 0.05:
        side = "lower" if q[:, i].min() - lo[i] < hi[i] - q[:, i].max() else "upper"
        print(f"  {c:20s} {np.degrees(q[:, i].min()):7.1f} .. {np.degrees(q[:, i].max()):7.1f}  ({side} bound {np.degrees(lo[i] if side == 'lower' else hi[i]):.1f}, {100 * d:.1f} %)")
hits = [(c, max(qd[:, i].max() / vhi[i], qd[:, i].min() / vlo[i])) for i, c in enumerate(names) if np.isfinite(vhi[i]) and vhi[i] > 0]
print("velocities above 80 % of the bound:", [(c, round(float(u), 2)) for c, u in hits if u > 0.8] or "none")
a = np.asarray(x.actuator_model[:n, np.asarray(m.actuator_model.idx["a"])])
mc = m.actuator_model.muscle_constants
vol = np.asarray(mc["fmax"]).ravel() * np.asarray(mc["lceopt"]).ravel()
w = vol / vol.sum()
share = (a ** 3 * w).mean(0)
share /= share.sum()
mus = m.actuator_model.names
def group(nm):
    k = re.sub(r"_[rl]$", "", nm)
    for key, pat in (("lumbar / abdominal wall", r"^(EO|IO|IL|LTpL|LTpT|MF|QL|PS|Ps|rect_abd|RA|IC|multif|rot|semisp|spin|Abs)"), ("thoracic / neck", r"^(TR|SPL|splen|supmult|ST|long_|stern|cleid|scalen|ExtIC|IntIC)"), ("shoulder / arm / hand", r"^(DELT|SUPSP|INFSP|SUBSC|TMIN|TMAJ|PECM|LAT|CORB|TRI|BIC|BRA|ANC|PRO|SUP|FCR|FCU|ECR|ECU|ED|EI|PL|FD|EP|AP|FP|EDM|RI|LU|UI|OP|APB|ADP)")):
        if re.match(pat, k):
            return key
    return "legs / other"
groups = defaultdict(float)
for nm, sh in zip(mus, share):
    groups[group(nm)] += sh
print("share of the muscle-effort term by region:", {k: f"{100 * v:.0f} %" for k, v in sorted(groups.items(), key=lambda kv: -kv[1])})
print("largest single contributions (share of effort, mean activation, peak, Fmax N, volume rank):")
for i in np.argsort(-share)[:10]:
    print(f"  {mus[i]:16s} {100 * share[i]:5.1f} %  a mean {a[:, i].mean():.2f} peak {a[:, i].max():.2f}  Fmax {np.asarray(mc['fmax']).ravel()[i]:6.0f}  L0 {1e3 * np.asarray(mc['lceopt']).ravel()[i]:5.0f} mm")
print("muscles with activation above 0.1 at some node: %d of %d; median activation %.3f" % ((a.max(0) > 0.1).sum(), a.shape[1], np.median(a)))
