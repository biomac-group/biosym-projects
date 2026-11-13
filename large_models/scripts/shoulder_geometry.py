"""
Compare the shoulder of SIPP, MS-Human-700 and MyoFullBody: the orientation of the right humerus relative to the thorax
(z-x-y decomposition as in the state prior) when single shoulder coordinates are moved from the default pose.

    uv run python scripts/shoulder_geometry.py
"""
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from arm_geometry import MODELS as ARM  # noqa: E402
from biosym.model.contact.body_kinematics import BodyKinematics  # noqa: E402
from biosym.model.model import BiosymModel  # noqa: E402
from objectives.prior_map import zxy_angles  # noqa: E402

THORAX = {"SIPP (gait3d)": "torso", "MS-Human-700": "thoracic12", "MyoFullBody": "lumbar1"}
MOVES = {
    "SIPP (gait3d)": ("arm_flex_r", "arm_add_r", "arm_rot_r"),
    "MS-Human-700": ("elv_angle_r", "shoulder_elv_r", "shoulder_rot_r"),
    "MyoFullBody": ("elv_angle_r", "shoulder_elv_r", "shoulder_rot_r"),
}


def main():
    for label in THORAX:
        path = ARM[label][0]
        m = BiosymModel(str(path), compile_eom=False, fit_muscle_paths=False)
        kin = BodyKinematics(m, bodies=(THORAX[label], "humerus_r"))
        consts = jnp.asarray(m.default_constants.filter("model").to_array())
        names = [n.removeprefix("q_") for n in m.coordinates.names]
        R0, _ = kin.poses(jnp.zeros(len(names)), consts)
        ti, hi = kin.index(THORAX[label]), kin.index("humerus_r")
        print(label)
        for coord in MOVES[label]:
            for deg in (-30, 30, 60):
                q = np.zeros(len(names))
                q[names.index(coord)] = np.radians(deg)
                R, _ = kin.poses(jnp.asarray(q), consts)
                A_t = np.asarray(R[ti]) @ np.asarray(R0[ti]).T
                A_h = np.asarray(R[hi]) @ np.asarray(R0[hi]).T
                a = np.degrees(np.asarray(zxy_angles(jnp.asarray(A_t.T @ A_h))))
                print(f"  {coord:15s}{deg:+4d} deg -> humerus rel. thorax z/x/y = {np.round(a, 1)}")
        # swing in the sagittal plane: plane of elevation 90 deg, elevation +-30 deg
        if label != "SIPP (gait3d)":
            for deg in (-30, 30):
                q = np.zeros(len(names))
                q[names.index("elv_angle_r")] = np.radians(90)
                q[names.index("shoulder_elv_r")] = np.radians(deg)
                R, _ = kin.poses(jnp.asarray(q), consts)
                A_t = np.asarray(R[ti]) @ np.asarray(R0[ti]).T
                A_h = np.asarray(R[hi]) @ np.asarray(R0[hi]).T
                a = np.degrees(np.asarray(zxy_angles(jnp.asarray(A_t.T @ A_h))))
                print(f"  elv_angle  90 + shoulder_elv{deg:+4d} deg -> humerus rel. thorax z/x/y = {np.round(a, 1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
