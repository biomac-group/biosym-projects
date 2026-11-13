"""
Bake the MS-Human-700 meshes and muscle lines into the body frames of the biosym model, for the biosym viewer.

    uv run --with "mujoco>=3.4" python scripts/make_visuals.py MODEL        # writes models/MODEL/visuals.npz

The viewer (``biosym-view``, ``Collocation.view``) draws ``visuals.npz`` when the model yaml names it
(``additional_parameters.visuals.file``). It holds

* every mesh geom of the MJCF (bones; the mirrored ones already mirrored by MuJoCo), with the vertices in the frame of
  the biosym body they belong to. Bodies without a joint were merged into their parent by ``make_model.py``, frames of
  bodies with hinge anchors were moved and the root re-expressed in its ``root_frame``; the same maps are applied here;
* every muscle as the polyline through its path sites, in the same body frames and in the actuator order (the names of
  ``muscle_names.csv``), so the viewer can colour it by the activation of the solution. Wrapping geometry is not drawn: the
  line runs straight between the sites, as the viewer does for OpenSim muscles.
"""
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_model as mm  # noqa: E402
from model_config import from_argv  # noqa: E402


def to_kept_frames(m, bodies):
    """
    Map every MuJoCo body onto the biosym body that carries it.

    Parameters
    ----------
    m : mujoco.MjModel
    bodies : dict
        Result of ``make_model.build_bodies`` (kept bodies with their MuJoCo id ``mj`` and frame ``frame_R``, ``frame_p``).

    Returns
    -------
        Dict body id -> ``(biosym body name, R, p)``: the rotation and the origin of the MuJoCo body frame in the biosym
        frame of the body that carries it (merged bodies are rigid in it; moved and re-expressed frames accounted for).
    """
    kept = {b["mj"]: name for name, b in bodies.items()}
    out = {}
    for b in range(1, m.nbody):
        R, p, c = np.eye(3), np.zeros(3), b
        while c not in kept:                                # walk up through the jointless bodies
            Rc = mm.quat_to_mat(m.body_quat[c])
            R, p = Rc @ R, Rc @ p + m.body_pos[c]
            c = m.body_parentid[c]
        k = bodies[kept[c]]
        out[b] = (kept[c], k["frame_R"].T @ R, k["frame_R"].T @ (p - k["frame_p"]))
    return out


def main():
    cfg = from_argv()
    OUT = cfg.out / "visuals.npz"
    m = mujoco.MjModel.from_xml_path(str(cfg.mjcf))
    bodies = mm.build_bodies(m, cfg.root_frame)
    frame = to_kept_frames(m, bodies)
    rename = {r["mujoco_name"]: r["name"] for r in csv.DictReader(open(cfg.names_csv))}

    mesh_names, mesh_bodies, verts, faces, v_end, f_end = [], [], [], [], [0], [0]
    for g in range(m.ngeom):
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        k = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].reshape(-1, 3)
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]].reshape(-1, 3)
        assert f.max() < len(v), "face indices are not local to the mesh"
        body, Rb, pb = frame[int(m.geom_bodyid[g])]
        Rg = mm.quat_to_mat(m.geom_quat[g])
        v_body = v @ Rg.T + m.geom_pos[g]                   # MuJoCo body frame (mesh_vert is in the geom's frame)
        mesh_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, k) or f"mesh{k}")
        mesh_bodies.append(body)
        verts.append((v_body @ Rb.T + pb).astype(np.float32))
        faces.append(f.astype(np.int32))
        v_end.append(v_end[-1] + len(v))
        f_end.append(f_end[-1] + len(f))

    muscle_names, point_body, points, p_end = [], [], [], [0]
    for a in range(m.nu):
        tendon = int(m.actuator_trnid[a, 0])
        adr, num = int(m.tendon_adr[tendon]), int(m.tendon_num[tendon])
        pts = []
        for w in range(adr, adr + num):
            if int(m.wrap_type[w]) != int(mujoco.mjtWrap.mjWRAP_SITE):
                continue
            site = int(m.wrap_objid[w])
            body, Rb, pb = frame[int(m.site_bodyid[site])]
            pts.append((body, Rb @ m.site_pos[site] + pb))
        muscle_names.append(rename[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)])
        point_body += [b for b, _ in pts]
        points += [p for _, p in pts]
        p_end.append(p_end[-1] + len(pts))

    np.savez_compressed(
        OUT, mesh_names=np.array(mesh_names), mesh_bodies=np.array(mesh_bodies), mesh_v_end=np.array(v_end),
        mesh_f_end=np.array(f_end), vertices=np.concatenate(verts), faces=np.concatenate(faces),
        muscle_names=np.array(muscle_names), muscle_end=np.array(p_end), point_bodies=np.array(point_body),
        points=np.array(points, dtype=np.float32))
    print(f"{len(mesh_names)} meshes ({v_end[-1]} vertices, {f_end[-1]} triangles), {len(muscle_names)} muscles "
          f"({p_end[-1]} path points) -> {OUT.name}, {OUT.stat().st_size / 1e6:.1f} MB")
    # sanity: extent of the bones in the biosym world at the default pose is the extent of a standing human
    return 0


if __name__ == "__main__":
    sys.exit(main())
