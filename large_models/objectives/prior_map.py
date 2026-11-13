"""
The learned state prior (biosym ``state_prior_term``) for the MuJoCo models of this project (MS-Human-700, MyoFullBody).

The prior scores the 23 coordinates of the SIPP / gait3d model. Both models have other coordinates for the trunk and the
arms (their legs and arms come from the same OpenSim models), so they are mapped (`PriorCoordinates`):

* hips and ankles: the same coordinates (OpenSim names and axes);
* knees: negated, both models count knee flexion positive and the prior expects the gait3d sign (flexion negative);
* lumbar extension / bending / rotation: the orientation of the thorax (the body that carries the ribcage: MS-Human's
  ``thoracic12``, MyoFullBody's ``lumbar1``, objective argument ``thorax``) relative to the pelvis, decomposed in the
  gait3d lumbar sequence (z, x, y; body-fixed). Both spines have several joints in series (MS-Human L5_S1 and T12_L1,
  MyoFullBody five lumbar levels coupled to three coordinates), so the map sums their effect exactly;
* arm flexion / adduction / rotation: the orientation of the humerus relative to the thorax in the gait3d shoulder
  sequence (z, x, y). The left side is mirrored first (z -> -z), since gait3d's left arm axes are mirrored (-x, -y);
* elbow flexion: the same coordinate (both axes along +z);
* pro_sup: ``pro_sup + PRO_SUP_OFFSET_DEG`` (114 deg). Measured, not assumed (``scripts/forearm_offset.py``): the
  thumb metacarpal's direction around the forearm axis, relative to the elbow axis, is the same in SIPP / gait3d and in
  MyoFullBody or MS-Human (identical forearms) when gait3d's ``pro_sup`` is the models' plus 114 deg (100-118 deg over
  the range, same direction). Neutral (thumb forward) is gait3d 81 deg, the models' -34 deg. An assumed +90 deg (both
  neutral at the default pose) made the prior see the forearms 24 deg too pronated.

Orientations are measured from the model's default pose, in which it stands upright with hanging arms, the pose whose
gait3d coordinates are all zero. Use it in a collocation yaml as ``name: objectives/prior_map.py`` (path relative to the
working directory) with ``args: {thorax: <body>}``, and pass ``PriorCoordinates(model, thorax)`` to
``make_prior_guess(coordinate_map=...)``.
"""
import jax.numpy as jnp
import numpy as np

from biosym.model.contact.body_kinematics import BodyKinematics
from biosym.objectives import state_prior_term

PELVIS = "pelvis"
PRO_SUP_OFFSET_DEG = 114.0      # gait3d pro_sup minus the models' pro_sup for the same forearm rotation (forearm_offset.py)
MIRROR = np.diag([1.0, 1.0, -1.0])


def zxy_angles(R):
    """
    Decompose a rotation matrix into body-fixed z, x, y angles, ``R = Rz(a) Rx(b) Ry(c)``.

    Parameters
    ----------
    R : jax.Array
        3x3 rotation matrix.

    Returns
    -------
        ``(a, b, c)`` in radians; ``b`` in (-pi/2, pi/2).
    """
    b = jnp.arcsin(jnp.clip(R[2, 1], -1.0 + 1e-12, 1.0 - 1e-12))
    a = jnp.arctan2(-R[0, 1], R[1, 1])
    c = jnp.arctan2(-R[2, 0], R[2, 2])
    return a, b, c


class PriorCoordinates:
    """
    Map one node's coordinates (MS-Human-700 or MyoFullBody) onto the 23 coordinates of the state prior (gait3d convention).

    Parameters
    ----------
    model : BiosymModel
        A model of this project.
    thorax : str
        Body that carries the ribcage (``thoracic12`` for MS-Human-700, ``lumbar1`` for MyoFullBody).
    """

    def __init__(self, model, thorax="thoracic12"):
        self.thorax = thorax
        self.names = [str(n).removeprefix("q_") for n in model.coordinates.names]
        # only these bodies and their ancestors are walked (20 of MyoFullBody's 69): all of them, fingers included,
        # made the gradient compile in 50 s instead of 16 s
        self.kin = BodyKinematics(model, bodies=(PELVIS, thorax, "humerus_r", "humerus_l"))
        self.consts = jnp.asarray(model.default_constants.filter("model").to_array())
        R0, _ = self.kin.poses(jnp.zeros(len(self.names)), self.consts)
        self.body = {b: self.kin.index(b) for b in (PELVIS, thorax, "humerus_r", "humerus_l")}
        self.R0 = {b: np.asarray(R0[i]) for b, i in self.body.items()}

    def __call__(self, q):
        """
        The 23 prior coordinates of the coordinate vector ``q`` (one node), in the prior's order.

        Parameters
        ----------
        q : jax.Array
            Coordinate vector of the model.

        Returns
        -------
            jax.Array of 23 angles [rad].
        """
        R, _ = self.kin.poses(q, self.consts)
        # each segment's rotation away from the default pose, in the ground frame of the default pose
        A = {b: R[i] @ self.R0[b].T for b, i in self.body.items()}
        lumbar = zxy_angles(A[PELVIS].T @ A[self.thorax])
        arm_r = zxy_angles(A[self.thorax].T @ A["humerus_r"])
        arm_l = zxy_angles(MIRROR @ (A[self.thorax].T @ A["humerus_l"]) @ MIRROR)
        c = lambda n: q[self.names.index(n)]
        leg = lambda s: [c(f"hip_flexion_{s}"), c(f"hip_adduction_{s}"), c(f"hip_rotation_{s}"), -c(f"knee_angle_{s}"),
                         c(f"ankle_angle_{s}")]
        arm = lambda s, angles: [*angles, c(f"elbow_flexion_{s}"), c(f"pro_sup_{s}") + jnp.radians(PRO_SUP_OFFSET_DEG)]
        return jnp.stack([*leg("r"), *leg("l"), *lumbar, *arm("r", arm_r), *arm("l", arm_l)])


class Objective(state_prior_term.Objective):
    """
    ``state_prior_term`` with the coordinate map of this project's models (`PriorCoordinates`).

    Parameters
    ----------
    thorax : str, optional
        Body that carries the ribcage; default ``thoracic12`` (MS-Human-700).
    """

    def __init__(self, model, settings, **kwargs):
        self._thorax = kwargs.pop("thorax", "thoracic12")
        super().__init__(model, settings, **kwargs)

    def _init_coordinates(self, model, prior):
        expected = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
                    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
                    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
                    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r",
                    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l"]
        if list(prior["coord_names"]) != expected:
            raise ValueError(f"ms_human_prior: the prior's coordinates changed: {prior['coord_names']}")
        self.cols = None
        self.mapper = PriorCoordinates(model, self._thorax)

    def coordinate_map(self, q):
        return self.mapper(q)
