"""
Compare the arm chain of the models in the world frame (x forward, y up, z right), all coordinates at zero but the elbow.

    uv run python scripts/arm_geometry.py

For each model the upper arm (shoulder to elbow) and the forearm (elbow to wrist) as unit vectors at elbow flexion 0, 45
and 90 deg, and the forearm's lateral deviation (z) as an angle. In a model whose elbow axis is perpendicular to the
sagittal plane the forearm stays in it (z = 0).
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from biosym.model.contact.body_kinematics import BodyKinematics  # noqa: E402
from biosym.model.model import BiosymModel  # noqa: E402

PROJECTS = Path(__file__).resolve().parents[2]
MODELS = {
    "SIPP (gait3d)": (PROJECTS / "3d_simulations/models/sipp_runmad/sipp_runmad.yaml", "hand_r", ("elbow_flex_r",)),
    "Rajagopal2016": (Path("/Users/markusgambietz/PhD/01_Python_Projects/smartphone_personalisation/data/reprocess_Nag2023/generic_rajogopal.osim"),
                      "hand_r", ("elbow_flex_r",)),
    "MS-Human-700": (PROJECTS / "large_models/models/ms_human_700/ms_human_700.yaml", "proximal_row_r", ("elbow_flexion_r",)),
    "MyoFullBody": (PROJECTS / "large_models/models/myofullbody/myofullbody.yaml", "lunate_r", ("elbow_flexion_r",)),
}


def main():
    for label, (path, wrist, elbow) in MODELS.items():
        m = BiosymModel(str(path), compile_eom=False, fit_muscle_paths=False)
        kin = BodyKinematics(m)
        consts = jnp.asarray(m.default_constants.filter("model").to_array())
        names = [n.removeprefix("q_") for n in m.coordinates.names]
        bodies = {b["name"] for b in m.dicts["bodies"]}
        assert wrist in bodies, f"{label}: no body {wrist}"
        print(f"{label}")
        for deg in (0, 45, 90):
            q = np.zeros(len(names))
            q[names.index(elbow[0])] = np.radians(deg)
            _, p = kin.poses(jnp.asarray(q), consts)
            p = np.asarray(p)
            upper = p[kin.index("ulna_r")] - p[kin.index("humerus_r")]
            fore = p[kin.index(wrist)] - p[kin.index("ulna_r")]
            u, f = upper / np.linalg.norm(upper), fore / np.linalg.norm(fore)
            lateral = np.degrees(np.arcsin(f[2]))
            print(f"  elbow {deg:3d}: upper arm {np.round(u, 3)} ({1e3 * np.linalg.norm(upper):.0f} mm), forearm {np.round(f, 3)} ({1e3 * np.linalg.norm(fore):.0f} mm), forearm lateral {lateral:+.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
