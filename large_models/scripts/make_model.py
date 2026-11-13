"""
Convert an MJCF model of this project into biosym model files (``models/<MODEL>/``).

    uv run --with "mujoco>=3.4" python scripts/make_model.py MODEL        # MODEL: ms_human_700 or myofullbody
                                                                          # 10 to 20 minutes, most of it the path fits

The models and their specifics are in ``model_config.py``. The MJCF is compiled with the ``mujoco`` package (includes and
default classes resolved) and written as files biosym's ``MujocoParser`` reads directly:

``<MODEL>.xml``
    The body tree with explicit values: ``pos``, ``quat`` and the inertia (``fullinertia`` about the centre of mass, body
    frame) of every body. Bodies without a joint (vertebrae between the spinal joints, ribs, sternum, sacrum, ...) are
    merged into their parent, with their mass, centre of mass and inertia combined exactly. A body whose hinges rotate
    about a point other than its origin (``jnt_pos``, MS-Human's knees: 9 mm) gets its frame moved to that point, since
    biosym rotates about the body origin. The root body is biosym's ``pelvis``: its frame is the MuJoCo body
    ``root_frame`` (the pelvis, x forward, y up, z right), its own place in the world is dropped, so the world is that
    frame in the default pose: y up (gravity ``0 -9.81 0``), floor at y = 0, and the default standing height is the
    MuJoCo height of the pelvis (``pelvis_ty`` about 0.95 m for MS-Human, 1.0 m for MyoFullBody). A free root joint
    (MyoFullBody) becomes ``pelvis_tx/ty/tz`` and the rotations ``pelvis_tilt`` (z), ``pelvis_list`` (x),
    ``pelvis_rotation`` (y). The joint couplings and locks (MuJoCo ``<equality><joint polycoef>``) are written as such.
    Joint damping and stiffness are not written: the passive joint model is in the yaml.
``<MODEL>_muscles.xml``
    One ``<mujoco_muscle>`` per actuator (biosym's ``MujocoMuscle``) with MuJoCo's parameters (``force``, ``lengthrange``,
    ``range``, ``lmin``, ``lmax``, ``vmax``, ``fpmax``, ``fvmax``, ``tact``, ``tdeact``) and its path as a polynomial of the
    coordinates it crosses (see `fit_path`), under the names of ``muscle_names.csv`` (``scripts/pair_muscles.py``). Where
    MuJoCo's passive force uses other parameters than the active one (``biasprm`` != ``gainprm``; two MyoFullBody
    muscles) they are written as ``bias_range``, ``bias_fmax``, ``bias_lmax``, ``bias_fpmax``.
``<MODEL>.yaml``
    The model wrapper: the files above, the foot contact (``contact_implicit.xml``) and the passive joint model of the
    3d_simulations standard (toolbox joint model: K1 = 1 N m/rad, B = 1 N m s/rad, quadratic walls of 5000 N m/rad^2
    outside the MuJoCo ranges; Falisse et al. 2022 toe spring, 25 N m/rad and 2 N m s/rad, no wall).
``fit_report.csv``
    Per muscle: the crossed coordinates, the number of terms, and the length and moment-arm errors of the fit on poses
    that were not fitted.

Path fit. A muscle crosses a coordinate when the derivative of its tendon length with respect to the coordinate exceeds
``CROSS_TOL`` at one of ``N_PROBE`` random poses (coupled joints follow their driver, so the derivative is the total
one). Its moment arms are sampled over the ranges of its crossed coordinates (the MuJoCo ranges, narrowed by the
model's ``fit_ranges_deg`` where MuJoCo's tendon lengths jump; a grid of ``GRID_STEPS`` per coordinate up to 3 coordinates, as the
BioMAC toolbox does, random poses beyond, ``SAMPLES_PER_COORD`` per coordinate), and ``-moment arm =
dL/dq`` is regressed on the derivatives of the monomials of total degree 1 to `max_degree` by orthogonal matching pursuit
(the toolbox's stepwise regression is too slow for torso muscles with up to 9 coordinates), up to ``MAX_TERMS`` terms or
until the RMS residual is ``REL_ERROR`` of the largest moment arm. The constant is the tendon length with the crossed
coordinates at zero, and outside the sampled range biosym continues the polynomial linearly.
"""
import csv
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fibre_types  # noqa: E402
from model_config import from_argv  # noqa: E402

CROSS_TOL = 1e-4         # m/rad or m/m: smaller length derivatives do not count as crossing
N_PROBE = 400            # random poses for the crossing test
GRID_STEPS = {1: 15, 2: 11, 3: 8}   # samples per coordinate of the grid (muscles crossing up to 3 coordinates)
SAMPLES_PER_COORD = 150  # random samples per crossed coordinate beyond 3
N_CHECK = 300            # poses of the fit check
MAX_TERMS = 60
REL_ERROR = 0.01
SEED = 0

# MuJoCo enums as plain ints (numpy ints from the model arrays do not compare equal to pybind enums in every order)
HINGE, SLIDE, FREE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_FREE)
EQ_JOINT, TRN_TENDON = int(mujoco.mjtEq.mjEQ_JOINT), int(mujoco.mjtTrn.mjTRN_TENDON)

# passive joint model of the 3d_simulations standard (sipp_runmad.yaml)
JOINT_DEFAULT = {"damping": 1.0, "spring_stiffness": 1.0, "spring_ref": 0.0, "limit_quadratic": 5000.0}
TOE = {"damping": 2.0, "spring_stiffness": 25.0, "spring_ref": 0.0, "limit_quadratic": 0.0}   # Falisse et al. 2022


def fmt(values):
    """Space separated text of numbers, at full precision."""
    return " ".join(f"{float(v):.12g}" for v in np.ravel(values))


def quat_to_mat(quat):
    """Rotation matrix of a MuJoCo quaternion (child frame -> parent frame)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def mat_to_quat(R):
    """MuJoCo quaternion of a rotation matrix."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(-1))
    return q


# ------------------------------------------------------------------------------------------------------------- kinematics
SIDE_TOKEN = re.compile(r"_([lr])(?=_|$)")      # biosym.constraints.periodicity._SIDE_TOKEN


def joint_names(m):
    """
    Joint names for biosym: MuJoCo's, with ``_r`` added to a right-side joint that has none while its ``_l`` partner exists.

    biosym's mirror pairs coordinates by their last ``_r``/``_l`` token; MyoFullBody names four right-hand joints without
    one (``cmc_flexion``, ``cmc_abduction``, ``md3_flexion``, ``pm4_flexion`` next to ``cmc_flexion_l``, ...).

    Returns
    -------
        List of names in joint order.
    """
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    present = set(names)
    return [n + "_r" if not SIDE_TOKEN.search(n) and n + "_l" in present else n for n in names]


class Coordinates:
    """
    The independent coordinates of the compiled model and the map to MuJoCo's ``qpos`` and velocity space.

    Hinges and slides that no ``<equality><joint>`` drives or locks are the independent coordinates. A free root joint
    is no coordinate here: it stays at its default (muscle paths do not depend on it), and the biosym model replaces it
    by the six pelvis coordinates.

    Parameters
    ----------
    m : mujoco.MjModel
        Compiled model; besides one free root joint, every joint is a hinge or a slide.
    fit_ranges_deg : dict, optional
        Coordinates whose paths are fitted on less than the MuJoCo range, ``{name: (lower, upper)}`` [deg].
    """

    def __init__(self, m, fit_ranges_deg=None):
        types = [int(t) for t in m.jnt_type]
        assert all(t in (HINGE, SLIDE) for t in types[1:]) and types[0] in (HINGE, SLIDE, FREE), "unsupported joints"
        self.m = m
        self.names = joint_names(m)
        self.free = [j for j in range(m.njnt) if types[j] == FREE]
        self.driven = {}       # joint -> (driver joint or None for a locked joint, polycoef)
        for e in range(m.neq):
            if int(m.eq_type[e]) == EQ_JOINT and m.eq_active0[e]:
                j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                self.driven[j1] = (j2 if j2 >= 0 else None, np.array(m.eq_data[e, :5]))
        assert not any(d in self.driven for d, _ in self.driven.values() if d is not None), "chained couplings"
        hinge_slide = [j for j in range(m.njnt) if types[j] != FREE]
        assert not np.any(m.qpos0[m.jnt_qposadr[hinge_slide]]), "joint reference positions must be zero"
        self.independent = [j for j in hinge_slide if j not in self.driven]
        self.ind_names = [self.names[j] for j in self.independent]
        self.range = np.array([m.jnt_range[j] for j in self.independent])
        # the range the paths are fitted on: the MuJoCo range, narrowed by fit_ranges_deg
        self.fit_range = self.range.copy()
        for name, (lo, hi) in (fit_ranges_deg or {}).items():
            k = self.ind_names.index(name)
            self.fit_range[k] = [max(self.range[k, 0], np.radians(lo)), min(self.range[k, 1], np.radians(hi))]
        self._qadr = m.jnt_qposadr[self.independent]
        self._vadr = m.jnt_dofadr[self.independent]

    @staticmethod
    def _poly(c, x):
        return c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4

    def qpos(self, q):
        """MuJoCo ``qpos`` of independent coordinates ``q`` (couplings and locks applied, a free root at its default)."""
        qpos = self.m.qpos0.copy()
        qpos[self._qadr] = q
        for j, (d, c) in self.driven.items():
            qpos[self.m.jnt_qposadr[j]] = c[0] if d is None else self._poly(c, qpos[self.m.jnt_qposadr[d]])
        return qpos

    def jacobian(self, q):
        """``d qvel / d qdot`` (nv x n_independent) at ``q``: the joint velocities of unit coordinate velocities."""
        qpos = self.qpos(q)
        J = np.zeros((self.m.nv, len(self.independent)))
        col = {j: k for k, j in enumerate(self.independent)}
        J[self._vadr, np.arange(len(self.independent))] = 1.0
        for j, (d, c) in self.driven.items():
            if d is None:
                continue
            x = qpos[self.m.jnt_qposadr[d]]
            J[self.m.jnt_dofadr[j], col[d]] = c[1] + 2 * c[2] * x + 3 * c[3] * x ** 2 + 4 * c[4] * x ** 3
        return J


def tendon_lengths_and_moments(m, d, coords, q):
    """
    Tendon lengths and their total derivatives with respect to the independent coordinates.

    Returns
    -------
        ``(ntendon,)`` lengths and ``(ntendon, n_independent)`` derivatives ``dL/dq``.
    """
    d.qpos[:] = coords.qpos(q)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_tendon(m, d)
    return d.ten_length.copy(), tendon_jacobian(m, d) @ coords.jacobian(q)


def tendon_jacobian(m, d):
    """Dense ``(ntendon, nv)`` tendon Jacobian (MuJoCo 4 keeps it sparse; the pattern is ``m.ten_J_rownnz/rowadr/colind``)."""
    J = np.asarray(d.ten_J).reshape(-1)
    if not hasattr(m, "ten_J_colind"):        # MuJoCo 3: dense
        return J.reshape(m.ntendon, m.nv)
    dense = np.zeros((m.ntendon, m.nv))
    for t in range(m.ntendon):
        adr, nnz = m.ten_J_rowadr[t], m.ten_J_rownnz[t]
        dense[t, m.ten_J_colind[adr:adr + nnz]] = J[adr:adr + nnz]
    return dense


# ------------------------------------------------------------------------------------------------------------- body tree
def build_bodies(m, root_frame="pelvis"):
    """
    Flatten the compiled body tree: merge bodies without joints into their parent and give every kept body its frame.

    The frame of a kept body is the MuJoCo body frame moved to its hinge anchor; the root's frame is the frame of the
    MuJoCo body ``root_frame`` (a jointless body merged into the root, or the root itself). With ``R_f``, ``p_f`` the
    biosym frame in the MuJoCo body frame, a point is ``x_mujoco = R_f x_biosym + p_f``.

    Parameters
    ----------
    m : mujoco.MjModel
    root_frame : str
        MuJoCo body whose frame becomes the root's.

    Returns
    -------
        Dict biosym body name -> ``{"mj": MuJoCo body id, "parent", "pos", "R", "mass", "com", "inertia" (3x3 about the
        com), "joints" (ids), "frame_R", "frame_p", "anchor", "members"}`` in parent-first order; ``pos``/``R`` relative to
        the parent's biosym frame, mass properties in the body's biosym frame. The root is named ``pelvis``.
    """
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    kept = [b for b in range(1, m.nbody) if m.body_jntnum[b] > 0]
    root = kept[0]
    assert m.body_parentid[root] == 0 and all(m.body_parentid[b] != 0 for b in kept[1:]), "one root body"

    def kept_ancestor(b):
        b = m.body_parentid[b]
        while b != 0 and m.body_jntnum[b] == 0:
            b = m.body_parentid[b]
        return b

    def pose_in(b, ancestor):
        """Pose of body b's frame in the frame of its ancestor (rigid chain of bodies at the default pose)."""
        R, p = np.eye(3), np.zeros(3)
        chain = []
        while b != ancestor:
            chain.append(b)
            b = m.body_parentid[b]
        for c in reversed(chain):
            Rc = quat_to_mat(m.body_quat[c])
            p = p + R @ m.body_pos[c]
            R = R @ Rc
        return R, p

    frame = {}
    for b in kept:
        hinges = [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
                  if int(m.jnt_type[j]) == HINGE]
        points = np.array([m.jnt_pos[j] for j in hinges]) if hinges else np.zeros((1, 3))
        assert np.allclose(points, points[0], atol=1e-9), f"{name(b)}: hinges with different anchors"
        frame[b] = (np.eye(3), points[0])
    rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_frame)
    assert rf == root or (m.body_jntnum[rf] == 0 and kept_ancestor(rf) == root), f"{root_frame} is not rigid with the root"
    assert not np.any(frame[root][1]), "the root's hinges must turn about its origin"
    fR, fp = pose_in(rf, root) if rf != root else (np.eye(3), np.zeros(3))
    # level the root frame: keep its heading, make y point against gravity (MyoFullBody's pelvis is tilted 0.8 deg in the
    # default pose; MS-Human's is level), so the biosym world is y up exactly
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    Rroot = d.xmat[root].reshape(3, 3)
    y = -np.asarray(m.opt.gravity) / np.linalg.norm(m.opt.gravity)
    x = (Rroot @ fR)[:, 0]
    x = (x - (x @ y) * y) / np.linalg.norm(x - (x @ y) * y)
    level = Rroot.T @ np.column_stack([x, y, np.cross(x, y)])
    tilt = np.degrees(np.arccos(np.clip((np.trace(fR.T @ level) - 1) / 2, -1, 1)))
    if tilt > 1e-6:
        print(f"root frame '{root_frame}' levelled by {tilt:.3f} deg (its y axis against gravity)")
    frame[root] = (level, fp)

    bodies = {}
    for b in kept:
        fR, fp = frame[b]
        members = [b] + [c for c in range(1, m.nbody) if m.body_jntnum[c] == 0 and kept_ancestor(c) == b]
        mass, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
        for c in members:
            Rc, pc = pose_in(c, b) if c != b else (np.eye(3), np.zeros(3))
            com = fR.T @ (pc + Rc @ m.body_ipos[c] - fp)                      # in the biosym frame
            Ri = fR.T @ Rc @ quat_to_mat(m.body_iquat[c])
            I_c = Ri @ np.diag(m.body_inertia[c]) @ Ri.T
            mass += m.body_mass[c]
            first += m.body_mass[c] * com
            second += I_c + m.body_mass[c] * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
        com = first / mass
        inertia = second - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
        parent = kept_ancestor(b)
        if parent == 0:              # the root: its place in the world defines the world of the biosym model
            R, p = np.eye(3), np.zeros(3)
        else:
            Rm, pm = pose_in(b, parent)
            fRp, fpp = frame[parent]
            R, p = fRp.T @ Rm @ fR, fRp.T @ (pm + Rm @ fp - fpp)
        bname = "pelvis" if b == root else name(b)
        bodies[bname] = {"mj": b, "parent": None if parent == 0 else ("pelvis" if parent == root else name(parent)),
                         "pos": p, "R": R, "mass": mass, "com": com, "inertia": inertia, "frame_R": fR, "frame_p": fp,
                         "anchor": fp, "joints": list(range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])),
                         "members": [name(c) for c in members]}
    return bodies


def biosym_world(m, bodies):
    """
    The biosym world in MuJoCo's world: rotation and origin of the root's biosym frame in the default pose.

    Returns
    -------
        ``(R0, p0)``: a point is ``x_biosym = R0^T (x_mujoco - p0)``; MuJoCo's gravity in biosym is ``R0^T g``.
    """
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    root = bodies["pelvis"]
    Rr = d.xmat[root["mj"]].reshape(3, 3)
    return Rr @ root["frame_R"], d.xpos[root["mj"]] + Rr @ root["frame_p"]


# pelvis coordinates that replace a free root joint: translations along the world axes, then the gait3d rotations
FREE_ROOT_JOINTS = [("pelvis_tx", "slide", (1, 0, 0), (-5, 5)), ("pelvis_ty", "slide", (0, 1, 0), (-1, 3)),
                    ("pelvis_tz", "slide", (0, 0, 1), (-3, 3)), ("pelvis_tilt", "hinge", (0, 0, 1), (-1.5708, 1.5708)),
                    ("pelvis_list", "hinge", (1, 0, 0), (-1.5708, 1.5708)),
                    ("pelvis_rotation", "hinge", (0, 1, 0), (-1.5708, 1.5708))]


def write_body_xml(m, bodies, coords, path, model_name):
    """Write the biosym-format MJCF of the flattened body tree and the joint couplings and locks."""
    jname = lambda j: coords.names[j]
    R0, _ = biosym_world(m, bodies)
    gravity = R0.T @ m.opt.gravity
    assert np.allclose(gravity, [0.0, -np.linalg.norm(m.opt.gravity), 0.0], atol=1e-6), f"root frame is not y up: {gravity}"
    root = ET.Element("mujoco", model=f"{model_name}_biosym")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", gravity=fmt([0.0, -np.linalg.norm(m.opt.gravity), 0.0]))
    world = ET.SubElement(root, "worldbody")
    elements = {None: world}
    for bname, b in bodies.items():
        attrib = {"name": bname, "pos": fmt(b["pos"])}
        if not np.allclose(b["R"], np.eye(3), atol=1e-12):
            attrib["quat"] = fmt(mat_to_quat(b["R"]))
        el = ET.SubElement(elements[b["parent"]], "body", attrib)
        I = b["inertia"]
        ET.SubElement(el, "inertial", pos=fmt(b["com"]), mass=f"{b['mass']:.12g}",
                      fullinertia=fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        joints = b["joints"]
        if any(int(m.jnt_type[j]) == FREE for j in joints):
            for jn, jt, axis, rng in FREE_ROOT_JOINTS:
                ET.SubElement(el, "joint", name=jn, type=jt, axis=fmt(axis), range=fmt(rng))
            joints = [j for j in joints if int(m.jnt_type[j]) != FREE]
        elif bname == "pelvis":      # translations first (they commute), in x, y, z order, then the rotations as given
            slides = sorted([j for j in joints if int(m.jnt_type[j]) == SLIDE], key=lambda j: jname(j))
            joints = slides + [j for j in joints if int(m.jnt_type[j]) == HINGE]
        for j in joints:
            ET.SubElement(el, "joint", name=jname(j),
                          type="slide" if int(m.jnt_type[j]) == SLIDE else "hinge",
                          axis=fmt(b["frame_R"].T @ m.jnt_axis[j]), range=fmt(m.jnt_range[j]))
        elements[bname] = el
    eq = ET.SubElement(root, "equality")
    for j, (dj, c) in coords.driven.items():
        attrib = {"joint1": jname(j), "polycoef": fmt(c)}
        if dj is not None:
            attrib["joint2"] = jname(dj)
        ET.SubElement(eq, "joint", attrib)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# ------------------------------------------------------------------------------------------------------------- path fits
def monomials(k, degree):
    """Exponents of all monomials of total degree 1..degree in k variables, ``(n, k)``, by degree."""
    out = [()]
    for _ in range(k):
        out = [e + (p,) for e in out for p in range(degree + 1)]
    rows = sorted((e for e in out if 0 < sum(e) <= degree), key=lambda e: (sum(e), e))
    return np.array(rows, dtype=int).reshape(-1, k)


def max_degree(k):
    """Total degree of the candidate monomials for k crossed coordinates (the toolbox's 4, more for few, less for many)."""
    return {1: 6, 2: 6, 3: 5}.get(k, 4 if k <= 5 else 3)


def derivative_design(X, exponents):
    """Rows ``d monomial / d q_j`` at the samples X ``(n, k)``, stacked over j: ``(n k, n_terms)``."""
    blocks = []
    for j in range(X.shape[1]):
        shifted = exponents.copy()
        shifted[:, j] = np.maximum(shifted[:, j] - 1, 0)
        blocks.append(exponents[None, :, j] * np.prod(X[:, None, :] ** shifted[None, :, :], axis=2))
    return np.concatenate(blocks)


def omp(A, b, max_terms, tol):
    """
    Orthogonal matching pursuit: pick the column most correlated with the residual, refit, until ``rms <= tol``.

    Returns
    -------
        Selected column indices and their least-squares coefficients.
    """
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = np.inf
    selected, coeffs, r = [], np.zeros(0), b.copy()
    while len(selected) < max_terms:
        score = np.abs(A.T @ r) / norms
        score[selected] = -1.0
        selected.append(int(np.argmax(score)))
        coeffs = np.linalg.lstsq(A[:, selected], b, rcond=None)[0]
        r = b - A[:, selected] @ coeffs
        if np.sqrt(np.mean(r ** 2)) <= tol:
            break
    return selected, coeffs


def polynomial(X, exponents, coeffs):
    """Value of the polynomial at the samples X ``(n, k)``."""
    return np.prod(X[:, None, :] ** exponents[None, :, :], axis=2) @ coeffs


def fit_path(m, d, coords, tendon, crossed, rng):
    """
    Fit the path polynomial of one tendon over the ranges of its crossed coordinates.

    Parameters
    ----------
    m, d : mujoco.MjModel, mujoco.MjData
    coords : Coordinates
    tendon : int
        Tendon index.
    crossed : list of int
        Positions of the crossed coordinates in ``coords.ind_names``.
    rng : numpy.random.Generator

    Returns
    -------
        ``(fit, report)``: ``fit`` with ``length0``, ``coords``, ``lower``, ``upper``, ``exponents``, ``coeffs``;
        ``report`` with the fit errors on separate random poses.
    """
    k = len(crossed)
    lo, hi = coords.fit_range[crossed, 0], coords.fit_range[crossed, 1]
    q = np.zeros(len(coords.ind_names))
    length0 = tendon_lengths_and_moments(m, d, coords, q)[0][tendon]
    if k == 0:
        return {"length0": length0, "coords": [], "lower": [], "upper": [], "exponents": np.zeros((0, 0), int),
                "coeffs": np.zeros(0)}, {"n_terms": 0, "length_rms_mm": 0.0, "length_max_mm": 0.0, "arm_rms_mm": 0.0}

    def sample(X):
        L, dL = [], []
        for x in X:
            q[:] = 0.0
            q[crossed] = x
            length, moments = tendon_lengths_and_moments(m, d, coords, q)
            L.append(length[tendon])
            dL.append(moments[tendon, crossed])
        return np.array(L), np.array(dL)

    if k <= 3:
        axes = [np.linspace(a, b, GRID_STEPS[k]) for a, b in zip(lo, hi)]
        X = np.stack([g.reshape(-1) for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    else:
        X = rng.uniform(lo, hi, size=(SAMPLES_PER_COORD * k, k))
    _, dL = sample(X)
    exponents = monomials(k, max_degree(k))
    A = derivative_design(X, exponents)
    b = dL.T.reshape(-1)
    chosen, coeffs = omp(A, b, MAX_TERMS, REL_ERROR * np.max(np.abs(b)))
    exponents = exponents[chosen]

    Xc = rng.uniform(lo, hi, size=(N_CHECK, k))
    Lc, dLc = sample(Xc)
    L_fit = length0 + polynomial(Xc, exponents, coeffs)
    dL_fit = (derivative_design(Xc, exponents) @ coeffs).reshape(k, -1).T
    report = {"n_terms": len(coeffs), "length_rms_mm": 1e3 * np.sqrt(np.mean((L_fit - Lc) ** 2)),
              "length_max_mm": 1e3 * np.max(np.abs(L_fit - Lc)), "arm_rms_mm": 1e3 * np.sqrt(np.mean((dL_fit - dLc) ** 2))}
    return {"length0": length0, "coords": [coords.ind_names[c] for c in crossed], "lower": lo, "upper": hi,
            "exponents": exponents, "coeffs": coeffs}, report


def crossed_coordinates(m, d, coords, rng):
    """For every tendon, the independent coordinates its length depends on (`CROSS_TOL` at `N_PROBE` random poses)."""
    biggest = np.zeros((m.ntendon, len(coords.ind_names)))
    for _ in range(N_PROBE):
        q = rng.uniform(coords.range[:, 0], coords.range[:, 1])
        biggest = np.maximum(biggest, np.abs(tendon_lengths_and_moments(m, d, coords, q)[1]))
    return [list(np.flatnonzero(row > CROSS_TOL)) for row in biggest]


def write_muscle_xml(m, names, fits, path, title):
    """Write the ``mujoco_muscle`` actuator file of biosym."""
    root = ET.Element("actuators", type="mujoco_muscle", source=f"{title}, scripts/make_model.py")
    for a in range(m.nu):
        g, bias, dyn = m.actuator_gainprm[a], m.actuator_biasprm[a], m.actuator_dynprm[a]
        assert g[2] > 0, f"muscle {names[a]}: force must be given (MuJoCo's automatic force scale is not supported)"
        el = ET.SubElement(root, "mujoco_muscle", name=names[a], fmax=f"{g[2]:.12g}",
                           lengthrange=fmt(m.actuator_lengthrange[a]), range=fmt(g[0:2]), lmin=f"{g[4]:.12g}",
                           lmax=f"{g[5]:.12g}", vmax=f"{g[6]:.12g}", fpmax=f"{g[7]:.12g}", fvmax=f"{g[8]:.12g}",
                           tact=f"{dyn[0]:.12g}", tdeact=f"{dyn[1]:.12g}", tausmooth=f"{dyn[2]:.12g}")
        if not np.allclose(bias[:9], g[:9]):        # MuJoCo's passive force reads biasprm: range, force, lmax, fpmax
            el.set("bias_range", fmt(bias[0:2]))
            el.set("bias_fmax", f"{bias[2]:.12g}")
            el.set("bias_lmax", f"{bias[5]:.12g}")
            el.set("bias_fpmax", f"{bias[7]:.12g}")
        el.set("fasttwitch", f"{fibre_types.fast_twitch(names[a]):.6g}")
        f = fits[a]
        ET.SubElement(el, "path", length0=f"{f['length0']:.12g}", coords=" ".join(f["coords"]), lower=fmt(f["lower"]),
                      upper=fmt(f["upper"]), exponents=" ".join(str(int(e)) for e in np.ravel(f["exponents"])),
                      coeffs=fmt(f["coeffs"]))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_yaml(coords, path, cfg, n_muscles):
    """Write the model yaml: files, contact, passive joint model."""
    lines = [
        f"# {cfg.title} for biosym, written by scripts/make_model.py -- do not edit, rerun the script.",
        f"#   * {cfg.name}.xml: body tree of the MJCF with jointless bodies merged, y up, joint couplings as <equality>",
        f"#   * {cfg.name}_muscles.xml: {n_muscles} MuJoCo muscles (rigid tendon) with polynomial paths fitted to the MuJoCo tendons",
        "#   * contact_implicit.xml: spheres approximating the foot geometry, implicit Hunt-Crossley (SIPP law)",
        "#   * joints: toolbox passive joint model (K1 1 N m/rad, B 1 N m s/rad, quadratic walls 5000 N m/rad^2 outside the",
        "#     MuJoCo ranges), toes: Falisse et al. 2022 spring (25 N m/rad, 2 N m s/rad, no wall)",
        "#     changed per joint (model_config.passive_overrides): " + (", ".join(f"{k} {v}" for k, v in cfg.passive_overrides.items()) or "none"),
        "model:",
        f"  name: {cfg.name}.xml",
        "  additional_parameters:",
        "    ground_contact:",
        "      replace_existing: true",
        "      file: contact_implicit.xml",
        "    actuators:",
        "      replace_existing: true",
        f"      file: {cfg.name}_muscles.xml",
        "      activation_dynamics: true      # MuJoCo's activation dynamics with an excitation state per muscle",
        "    visuals:",
        "      file: visuals.npz              # meshes and muscle lines for the viewer (scripts/make_visuals.py)",
        "    joints:",
    ]
    for name, (lo, hi) in zip(coords.ind_names, coords.range):
        if name.startswith("pelvis_"):
            continue
        p = TOE if name.startswith("mtp_angle") else {**JOINT_DEFAULT, "passive_lower": lo, "passive_upper": hi}
        p = {**p, **cfg.passive_overrides.get(name, {})}
        lines.append(f"      {name}:")
        lines += [f"        {key}: {float(value):.10g}" for key, value in p.items()]
    path.write_text("\n".join(lines) + "\n")


def main():
    cfg = from_argv()
    out = cfg.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = mujoco.MjModel.from_xml_path(str(cfg.mjcf))
    d = mujoco.MjData(m)
    coords = Coordinates(m, cfg.fit_ranges_deg)
    with open(cfg.names_csv) as f:
        rename = {r["mujoco_name"]: r["name"] for r in csv.DictReader(f)}
    names = [rename[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)] for a in range(m.nu)]
    assert all(int(m.actuator_trntype[a]) == TRN_TENDON for a in range(m.nu))

    bodies = build_bodies(m, cfg.root_frame)
    write_body_xml(m, bodies, coords, out / f"{cfg.name}.xml", cfg.name)
    locked = sum(1 for dj, _ in coords.driven.values() if dj is None)
    n_coords = len(coords.ind_names) + (6 if coords.free else 0)       # a free root becomes the six pelvis coordinates
    print(f"bodies: {m.nbody - 1} -> {len(bodies)} (jointless bodies merged), mass "
          f"{sum(b['mass'] for b in bodies.values()):.4f} kg (MJCF {m.body_mass.sum():.4f}), {n_coords} coordinates, "
          f"{len(coords.driven) - locked} coupled and {locked} locked joints")
    moved = {n: np.round(b["anchor"], 6).tolist() for n, b in bodies.items() if n != "pelvis" and np.any(b["anchor"])}
    print(f"frames moved to the hinge anchor: {moved}")

    rng = np.random.default_rng(SEED)
    crossed = crossed_coordinates(m, d, coords, rng)
    counts = np.bincount([len(c) for c in crossed])
    print(f"crossed coordinates per muscle: {dict(enumerate(counts.tolist()))} ({time.time() - t0:.0f} s)", flush=True)
    fits, reports = [], []
    for a in range(m.nu):
        tendon = m.actuator_trnid[a, 0]
        fit, report = fit_path(m, d, coords, tendon, crossed[tendon], rng)
        fits.append(fit)
        reports.append({"muscle": names[a], "crossed": " ".join(fit["coords"]), **report})
        if (a + 1) % 50 == 0:
            print(f"  fitted {a + 1}/{m.nu} ({time.time() - t0:.0f} s)", flush=True)
    write_muscle_xml(m, names, fits, out / f"{cfg.name}_muscles.xml", cfg.title)
    with open(out / "fit_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reports[0]))
        w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in reports])
    lr = np.array([r["length_rms_mm"] for r in reports])
    lm = np.array([r["length_max_mm"] for r in reports])
    print(f"path fits: length RMS median {np.median(lr):.3f} mm, max {lr.max():.3f} mm; worst single error {lm.max():.3f} mm "
          f"({reports[int(np.argmax(lm))]['muscle']}); terms median {np.median([r['n_terms'] for r in reports]):.0f}")
    write_yaml(coords, out / f"{cfg.name}.yaml", cfg, m.nu)
    print(f"written to {out}: {cfg.name}.xml, {cfg.name}_muscles.xml, {cfg.name}.yaml, fit_report.csv ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
