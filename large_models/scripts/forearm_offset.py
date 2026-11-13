"""
Measure the offset between the models' ``pro_sup`` and gait3d's (SIPP), for ``objectives/prior_map.py``.

    uv run python scripts/forearm_offset.py          # needs ../3d_simulations (the SIPP model with its Geometry)

The same anatomical quantity in both models: the direction of the thumb metacarpal (mesh centroid) around the forearm
axis (elbow centre to wrist centre), relative to the elbow flexion axis made perpendicular to it (lateral); 0 is thumb
anterior (neutral), positive lateral (supinated). It is swept over each model's ``pro_sup`` with all other coordinates at
zero, and for each MyoFullBody value the SIPP value with the same thumb angle is found.
"""
import sys
from pathlib import Path

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from biosym.model.model import BiosymModel
from biosym.model.contact.body_kinematics import BodyKinematics
from biosym.results import viewer3d
from biosym.utils import useful_functions as uf

PROJECTS = Path(__file__).resolve().parents[2]

def setup(yaml, wrist_body, thumb_mesh):
    m = BiosymModel(yaml, compile_eom=False, fit_muscle_paths=False)
    meshes, _ = viewer3d.model_geometry(m)
    thumb = next(x for x in meshes if x["name"] == thumb_mesh)
    kin = BodyKinematics(m); c = jnp.asarray(m.default_constants.filter("model").to_array())
    names = [n.removeprefix("q_") for n in m.coordinates.names]
    ulna = next(b for b in m.dicts["bodies"] if b["name"] == "ulna_r")
    j = next(j for j in m.dicts["joints"] if j["name"] in ("elbow_flex_r", "elbow_flexion_r"))
    def angle(q):
        R, p = kin.poses(jnp.asarray(q), c); R, p = np.asarray(R), np.asarray(p)
        Rp = R[kin.index(ulna["parent"])] @ uf.rotation_matrix_xyz(np.asarray(ulna.get("body_orientation") or [0, 0, 0]))
        e = Rp @ (np.asarray(j["axis"]) / np.linalg.norm(j["axis"]))
        if e[2] < 0: e = -e
        elbow, wrist = p[kin.index("ulna_r")], p[kin.index(wrist_body)]
        f = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        lat = e - (e @ f) * f; lat /= np.linalg.norm(lat)
        ant = np.cross(lat, f)
        tb = kin.index(thumb["body"])
        t = (R[tb] @ thumb["vertices"].astype(float).mean(0) + p[tb]) - wrist
        return np.degrees(np.arctan2(t @ lat, t @ ant)), ant
    return m, names, angle

sipp, sn, sipp_angle = setup(str(PROJECTS / "3d_simulations/models/sipp_runmad/sipp_runmad.yaml"), "hand_r", "metacarpal1_rvs")
myo, mn, myo_angle = setup(str(PROJECTS / "large_models/models/myofullbody/myofullbody.yaml"), "lunate_r", "1mc_r")
msh, hn, msh_angle = setup(str(PROJECTS / "large_models/models/ms_human_700/ms_human_700.yaml"), "proximal_row_r", "hand_1mc")
print("thumb angle around the forearm axis (0 = thumb anterior, + = lateral / supinated), elbow straight:")
for v in (0, 30, 60, 90, 120, 150):
    q = np.zeros(len(sn)); q[sn.index("pro_sup_r")] = np.radians(v)
    a, ant = sipp_angle(q); print(f"  SIPP        pro_sup {v:+5d} deg -> {a:+7.1f} deg   (anterior axis {np.round(ant, 2)})")
for name, nm, fn in (("MyoFullBody", mn, myo_angle), ("MS-Human", hn, msh_angle)):
    for v in (-90, -45, 0, 45, 90):
        q = np.zeros(len(nm)); q[nm.index("pro_sup_r")] = np.radians(v)
        a, ant = fn(q); print(f"  {name:11s} pro_sup {v:+5d} deg -> {a:+7.1f} deg")
grid = np.radians(np.linspace(-10, 160, 171))
sa = []
for v in grid:
    q = np.zeros(len(sn)); q[sn.index("pro_sup_r")] = v; sa.append(sipp_angle(q)[0])
sa = np.array(sa)                                            # decreasing in pro_sup
offs = []
for v in np.radians(np.linspace(-80, 60, 15)):
    q = np.zeros(len(mn)); q[mn.index("pro_sup_r")] = v
    a = myo_angle(q)[0]
    theta = np.interp(-a, -sa, grid)                         # SIPP pro_sup with the same thumb angle
    offs.append(np.degrees(theta - v))
print(f"gait3d pro_sup - MyoFullBody pro_sup for equal forearm rotation: mean {np.mean(offs):.1f} deg, range {min(offs):.1f} to {max(offs):.1f} deg")
