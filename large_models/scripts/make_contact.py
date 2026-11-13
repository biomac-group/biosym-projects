"""
Write the foot contact of a model: spheres at the ends of the MJCF's foot capsules, levelled, with the SIPP contact law.

    uv run --with "mujoco>=3.4" python scripts/make_contact.py MODEL        # writes models/MODEL/contact_implicit.xml

Every collidable capsule of the bodies ``calcn_*`` and ``toes_*`` gives two spheres, at its end centres, with its radius
(the capsule's ends are where it meets the floor). The lowest points of the spheres are then levelled in the default
pose: each sphere is moved along the body's vertical by the difference to the lowest point of the heel spheres, so a
flat foot touches with heel and forefoot at once (MS-Human-700's hand-made spheres did not, and its standing solution
stood on the toes). The shifts are printed; for MyoFullBody they are at most a few millimetres.

The contact law is the SIPP standard's (``3d_simulations/models/sipp_runmad/contact_implicit.xml``): Hunt-Crossley,
implicit, stiffness 1e6, dissipation 2, friction 0.8, applied at the sphere centre. Not used for MS-Human-700, whose
contact file is hand-made (see MISSING.md).
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_model as mm  # noqa: E402
from model_config import from_argv  # noqa: E402

DEFAULT = {"name": "default", "body": "default", "pos": "0 0 0", "stiffness": "1e+06", "dissipation": "2",
           "static_friction": "0.8", "dynamic_friction": "0.8", "viscous_friction": "0.5", "transition_velocity": "0.2",
           "eps_depth": "0.001"}


def foot_spheres(m, bodies):
    """
    Spheres at the capsule ends of the feet, levelled to the heel in the default pose.

    Returns
    -------
        List of ``(name, body, centre in the biosym body frame, radius, shift [m])``, right foot first.
    """
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    R0, _ = mm.biosym_world(m, bodies)
    up = R0[:, 1]
    out = []
    for side in "rl":
        spheres = []
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            if body not in (f"calcn_{side}", f"toes_{side}") or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                continue
            b = bodies[body]
            r, half = m.geom_size[g, 0], m.geom_size[g, 1]
            axis = mm.quat_to_mat(m.geom_quat[g])[:, 2]
            Rb = d.xmat[m.geom_bodyid[g]].reshape(3, 3)
            for k, s in enumerate((-1, 1)):
                local = m.geom_pos[g] + s * half * axis                                # MuJoCo body frame
                low = float((d.xpos[m.geom_bodyid[g]] + Rb @ local) @ up - r)          # lowest point, world height
                gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                spheres.append([f"{gname}_{'ab'[k]}", body, b["frame_R"].T @ (local - b["frame_p"]), r, low,
                                b["frame_R"].T @ Rb.T @ up])
        heel = min(s[4] for s in spheres if s[1].startswith("calcn"))
        for name, body, centre, r, low, body_up in spheres:
            shift = heel - low
            out.append((name, body, centre + shift * body_up, r, shift))
    return out


def main():
    cfg = from_argv()
    m = mujoco.MjModel.from_xml_path(str(cfg.mjcf))
    bodies = mm.build_bodies(m, cfg.root_frame)
    spheres = foot_spheres(m, bodies)
    root = ET.Element("ground_contact_model", force_law="huntcrossley", implicit="true", application_point="center")
    root.insert(0, ET.Comment(f" Written by scripts/make_contact.py from the foot capsules of {cfg.title}: spheres at the"
                              " capsule ends, levelled to the heel in the default pose; SIPP contact law. "))
    ET.SubElement(ET.SubElement(root, "default"), "contact_point", DEFAULT)
    for name, body, centre, r, shift in spheres:
        ET.SubElement(root, "geometry", type="ContactSphere", name=name, body=body, pos=mm.fmt(centre),
                      radius=f"{r:.6g}")
        print(f"  {name:18s} {body:8s} radius {1e3 * r:4.1f} mm, levelled by {1e3 * shift:+5.2f} mm")
    ET.indent(root)
    path = cfg.out / "contact_implicit.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"{len(spheres)} spheres -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
