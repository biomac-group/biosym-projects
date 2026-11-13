"""
Check a converted biosym model against MuJoCo itself.

    uv run --with "mujoco>=3.4" python scripts/check_model.py MODEL        # MODEL: ms_human_700 or myofullbody

At random poses inside the joint ranges the paths are fitted on (and random velocities, accelerations and activations):

1. forward kinematics: every body's origin and orientation (biosym ``BodyKinematics`` vs MuJoCo ``xpos``/``xmat``, in the
   biosym world ``x_biosym = R0^T (x_mujoco - p0)``, see ``make_model.biosym_world``; a body is compared in its biosym
   frame: moved to its hinge anchor, the root re-expressed in its ``root_frame``);
2. inverse dynamics: biosym's RNEA torques against MuJoCo's ``mj_rne`` (``M qacc + bias``, no armature) projected onto
   biosym's coordinates, ``J^T (M (J qdd + J' qd) + bias)``, with the couplings' second derivative in ``J' qd``;
3. muscles: biosym's ``MujocoMuscle`` joint torques against MuJoCo's ``qfrc_actuator`` projected the same way, with
   MuJoCo's own tendon geometry, so this checks the path fits, the muscle curves and the parameter transfer at once;
4. mirror: the periodicity pairing with the model's ``mirror_negate`` reflects every body (biosym only);
5. visuals: the meshes and muscle points of ``visuals.npz`` against MuJoCo's world positions.

A free root joint (MyoFullBody) has no MuJoCo coordinates of biosym's ``pelvis_*``; its rotations are held at zero (pose
and velocity, accelerations random), where its velocities map linearly onto MuJoCo's: translations along the biosym world
axes (``R0``, world frame in MuJoCo), rotations about the root frame's axes (``frame_R``, body frame in MuJoCo).
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_model as mm  # noqa: E402
from model_config import from_argv  # noqa: E402
from biosym.constraints.periodicity import _mirror_name, get_symmetry_signs  # noqa: E402
from biosym.model.contact.body_kinematics import BodyKinematics  # noqa: E402
from biosym.model.model import BiosymModel  # noqa: E402
from biosym.utils.aba import get_rnea_biosym  # noqa: E402

N_POSES = 20
PELVIS = ("pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list", "pelvis_rotation")


class StateMap:
    """
    Biosym coordinates <-> MuJoCo ``qpos``/``qvel``.

    Parameters
    ----------
    m : mujoco.MjModel
    coords : make_model.Coordinates
    bodies : dict
        ``make_model.build_bodies``.
    names : list of str
        Biosym coordinate names in model order.
    """

    def __init__(self, m, coords, bodies, names):
        self.m, self.coords, self.names = m, coords, names
        self.R0, self.p0 = mm.biosym_world(m, bodies)
        self.free = bool(coords.free)
        root = bodies["pelvis"]
        if self.free:
            assert not np.any(root["frame_p"]), "a free root with an offset root frame is not supported here"
            self.root_dof = m.jnt_dofadr[coords.free[0]]
            self.root_qpos = m.jnt_qposadr[coords.free[0]]
            self.fR = root["frame_R"]
        self.ind_cols = [names.index(n) for n in coords.ind_names]           # biosym column of each MuJoCo coordinate
        self.pelvis_cols = [names.index(n) for n in PELVIS] if self.free else []

    def rotations_free(self):
        """Biosym columns that must stay at zero angle and velocity (the free root's rotations)."""
        return self.pelvis_cols[3:]

    def qpos(self, q):
        """MuJoCo ``qpos`` of biosym coordinates ``q`` (free root: its rotations must be zero)."""
        qpos = self.coords.qpos(q[self.ind_cols])
        if self.free:
            assert np.allclose(q[self.pelvis_cols[3:]], 0.0)
            qpos[self.root_qpos:self.root_qpos + 3] = self.m.qpos0[self.root_qpos:self.root_qpos + 3] + self.R0 @ q[self.pelvis_cols[:3]]
        return qpos

    def jacobian(self, q):
        """``d qvel / d qdot`` (nv x n_biosym) at ``q``."""
        J = np.zeros((self.m.nv, len(self.names)))
        J[:, self.ind_cols] = self.coords.jacobian(q[self.ind_cols])
        if self.free:
            r = self.root_dof
            J[r:r + 3, self.pelvis_cols[:3]] = self.R0                                   # world linear velocity
            # (tilt, list, rotation) turn about the root frame's z, x, y; MuJoCo's angular velocity is in the body frame
            J[r + 3:r + 6, self.pelvis_cols[3]] = self.fR[:, 2]
            J[r + 3:r + 6, self.pelvis_cols[4]] = self.fR[:, 0]
            J[r + 3:r + 6, self.pelvis_cols[5]] = self.fR[:, 1]
        return J

    def curvature(self, q, qd):
        """``J' qd``: the couplings' second derivative (the free root's rotations are at rest)."""
        qpos = self.qpos(q)
        qvel = self.jacobian(q) @ qd
        out = np.zeros(self.m.nv)
        for j, (dj, c) in self.coords.driven.items():
            if dj is None:
                continue
            x, xd = qpos[self.m.jnt_qposadr[dj]], qvel[self.m.jnt_dofadr[dj]]
            out[self.m.jnt_dofadr[j]] = (2 * c[2] + 6 * c[3] * x + 12 * c[4] * x ** 2) * xd ** 2
        return out


def main():
    cfg = from_argv()
    m = mujoco.MjModel.from_xml_path(str(cfg.mjcf))
    m.dof_armature[:] = 0.0
    d = mujoco.MjData(m)
    coords = mm.Coordinates(m, cfg.fit_ranges_deg)
    bodies = mm.build_bodies(m, cfg.root_frame)
    model = BiosymModel(str(cfg.out / f"{cfg.name}.yaml"))
    names = [str(n).removeprefix("q_") for n in model.coordinates.names]
    smap = StateMap(m, coords, bodies, names)
    R0, p0 = smap.R0, smap.p0
    lo, hi = np.zeros(len(names)), np.zeros(len(names))
    lo[smap.ind_cols], hi[smap.ind_cols] = coords.fit_range[:, 0], coords.fit_range[:, 1]
    for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
        lo[names.index(n)], hi[names.index(n)] = -0.3, 0.3
    still = smap.rotations_free()
    rng = np.random.default_rng(3)
    consts = model.default_constants
    kin = BodyKinematics(model)
    mj_of = {name: b["mj"] for name, b in bodies.items()}
    poses = jax.jit(lambda q: kin.poses(q, consts.filter("model").to_array()))
    rnea = jax.jit(lambda s: get_rnea_biosym(model)(s, consts))
    muscles = model.actuator_model
    torque = jax.jit(lambda s: muscles.forward(s, consts, model))
    forces = jax.jit(lambda s: muscles.muscle_equations(s))
    lengths = jax.jit(lambda s: muscles._length_and_moment_arm(s.q[:, None])[0])
    fmax = np.asarray(muscles.muscle_constants["fmax"]).reshape(-1)
    print(f"biosym model: {len(names)} coordinates, {len(kin.body_names)} bodies, {muscles.n_actuators} muscles, "
          f"total mass {float(np.sum(consts.mass)):.4f} kg (MJCF {m.body_mass.sum():.4f})")

    def mj_pose(name):
        b = bodies[name]
        Rm = d.xmat[b["mj"]].reshape(3, 3)
        return R0.T @ Rm @ b["frame_R"], R0.T @ (d.xpos[b["mj"]] + Rm @ b["frame_p"] - p0)

    err_p, err_R, err_tau, rel_tau, err_mus, rel_mus, err_F, err_len = [], [], [], [], [], [], [], []
    for _ in range(N_POSES):
        q = rng.uniform(lo, hi)
        qd, qdd = rng.normal(0, 1.0, len(q)), rng.normal(0, 5.0, len(q))
        q[still] = qd[still] = 0.0
        a = rng.uniform(0, 1, m.nu)

        # 1. kinematics
        d.qpos[:] = smap.qpos(q)
        mujoco.mj_kinematics(m, d)
        R, p = poses(jnp.asarray(q))
        for i, b in enumerate(kin.body_names):
            Rb, pb = mj_pose(b)
            err_p.append(np.linalg.norm(np.asarray(p[i]) - pb))
            err_R.append(np.abs(np.asarray(R[i]) - Rb).max())

        # 2. inverse dynamics
        J = smap.jacobian(q)
        d.qvel[:] = J @ qd
        d.act[:] = a
        mujoco.mj_forward(m, d)                  # positions, velocities, actuator forces (it overwrites qacc)
        d.qacc[:] = J @ qdd + smap.curvature(q, qd)
        result = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 1, result)
        tau_mj = J.T @ result
        state = model.default_states.replace(q=jnp.asarray(q), qd=jnp.asarray(qd), qdd=jnp.asarray(qdd))
        tau = np.asarray(rnea(state)).reshape(-1)
        err_tau.append(np.abs(tau - tau_mj).max())
        rel_tau.append(np.abs(tau - tau_mj).max() / np.abs(tau_mj).max())

        # 3. muscles (mj_forward above, with these activations)
        mus_mj = J.T @ d.qfrc_actuator
        block = np.zeros(muscles.get_n_states())
        block[np.asarray(muscles.idx["a"])] = a
        node = model.default_states.replace(q=jnp.asarray(q), qd=jnp.asarray(qd), actuator_model=jnp.asarray(block))
        mus = np.asarray(torque(node)).reshape(-1)
        err_mus.append(np.abs(mus - mus_mj).max())
        rel_mus.append(np.linalg.norm(mus - mus_mj) / np.linalg.norm(mus_mj))
        err_F.append(np.abs(np.asarray(forces(node)[1]).reshape(-1) + d.actuator_force) / fmax)
        err_len.append(np.abs(np.asarray(lengths(node)).reshape(-1) - d.actuator_length))

    print(f"1. kinematics: body origins max error {1e3 * max(err_p):.2e} mm, orientations max error {max(err_R):.2e}")
    print(f"2. inverse dynamics: max |tau - tau_mujoco| {max(err_tau):.2e} N m (relative to max |tau|: {max(rel_tau):.2e})")
    print(f"3. muscle torques: max |tau - tau_mujoco| {max(err_mus):.2f} N m, relative (2-norm) median "
          f"{np.median(rel_mus):.3f}, max {max(rel_mus):.3f}")
    err_F, err_len = np.array(err_F), np.array(err_len)
    worst = np.argsort(-err_F.max(axis=0))[:8]
    print(f"   muscle forces: |F - F_mujoco| / fmax median {np.median(err_F):.2e}, 99th percentile "
          f"{np.percentile(err_F, 99):.3f}; muscle-tendon length error median {1e3 * np.median(err_len):.3f} mm")
    print("   worst muscles (max force error / fmax, max length error mm): "
          + ", ".join(f"{muscles.names[i]} {err_F[:, i].max():.2f}/{1e3 * err_len[:, i].max():.1f}" for i in worst))

    # 4. mirror: q' = sign * q[perm] must put every body at the reflection (z -> -z) of its partner
    negate = list(cfg.mirror_negate)
    perm = [names.index(_mirror_name(n)) if _mirror_name(n) in names else i for i, n in enumerate(names)]
    sign = np.asarray(get_symmetry_signs(model, negate))[:len(names)]
    partner = [kin.body_names.index(_mirror_name(b)) if _mirror_name(b) in kin.body_names else i
               for i, b in enumerate(kin.body_names)]
    worst_mirror, worst_body = 0.0, None
    for _ in range(N_POSES):
        q = rng.uniform(lo, hi)
        _, p = poses(jnp.asarray(q))
        _, p_m = poses(jnp.asarray(sign * q[perm]))
        err = np.abs(np.asarray(p_m) - np.asarray(p)[partner] * np.array([1.0, 1.0, -1.0])).max(axis=1)
        if err.max() > worst_mirror:
            worst_mirror, worst_body = float(err.max()), kin.body_names[int(np.argmax(err))]
    print(f"4. mirror (mirror_negate {negate}): max body position error {1e3 * worst_mirror:.3f} mm ({worst_body})")

    # 5. visuals: baked meshes and muscle points moved by biosym's kinematics vs MuJoCo's world positions
    visuals = cfg.out / "visuals.npz"
    if not visuals.exists():
        print("5. visuals: no visuals.npz (scripts/make_visuals.py)")
        return 0
    from biosym.results.viewer3d import load_model_visuals
    meshes, muscle_lines = load_model_visuals(visuals)
    mesh_geoms = [g for g in range(m.ngeom) if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]
    sites_of = []                                  # MuJoCo site ids along each tendon, in actuator order
    for a_ in range(m.nu):
        t_ = int(m.actuator_trnid[a_, 0])
        sites_of.append([int(m.wrap_objid[w]) for w in range(m.tendon_adr[t_], m.tendon_adr[t_] + m.tendon_num[t_])
                         if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_SITE)])
    err_mesh = err_pt = 0.0
    for _ in range(3):
        q = rng.uniform(lo, hi)
        q[still] = 0.0
        d.qpos[:] = smap.qpos(q)
        mujoco.mj_kinematics(m, d)
        R, p = poses(jnp.asarray(q))
        R, p = np.asarray(R), np.asarray(p)
        for mesh, g in zip(meshes, mesh_geoms):
            i = kin.body_names.index(mesh["body"])
            ours = mesh["vertices"].astype(float) @ R[i].T + p[i]
            k = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].reshape(-1, 3)
            theirs = (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T - p0) @ R0
            err_mesh = max(err_mesh, float(np.abs(ours - theirs).max()))
        for line, sites in zip(muscle_lines, sites_of):
            for point, body, s in zip(line["points"], line["bodies"], sites):
                i = kin.body_names.index(body)
                err_pt = max(err_pt, float(np.abs(R[i] @ point + p[i] - (d.site_xpos[s] - p0) @ R0).max()))
    print(f"5. visuals: {len(meshes)} meshes max vertex error {1e3 * err_mesh:.3e} mm, {len(muscle_lines)} muscle lines "
          f"max path point error {1e3 * err_pt:.3e} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
