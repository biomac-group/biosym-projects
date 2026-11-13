"""
The models of this project and what the converter scripts need to know about each.

Every script takes the model name as its first argument (``ms_human_700`` or ``myofullbody``) and reads its
`ModelConfig` from `MODELS`.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelConfig:
    """
    What the converter needs to know about one MJCF model.

    Attributes
    ----------
    name : str
        Model name; also the stem of the written files (``<name>.xml``, ``<name>.yaml``, ``<name>_muscles.xml``).
    title : str
        Human-readable name for file headers.
    mjcf : Path
        The source MJCF.
    root_frame : str
        MuJoCo body whose frame becomes the frame of biosym's root body ``pelvis`` (x forward, y up, z right). The root's
        own frame is used when it is that body; otherwise (MyoFullBody's root ``Full Body`` is z up) the root is
        re-expressed in it.
    fit_ranges_deg : dict
        Coordinates whose muscle paths are fitted on less than the MuJoCo range, ``{name: (lower, upper)}`` in degrees.
    thorax : str
        Biosym body that carries the ribcage (the lumbar angles of the state prior are its orientation to the pelvis).
    mirror_negate : tuple of str
        Unpaired coordinates that change sign under the left/right mirror (periodicity ``mirror_negate``).
    viewer_port : int
        Port of the viewer ``run.py`` opens at the end (one per model, so both can stay open).
    passive_overrides : dict
        Joint name -> passive parameters that replace the toolbox defaults in the model yaml.
    """

    name: str
    title: str
    mjcf: Path
    root_frame: str = "pelvis"
    fit_ranges_deg: dict = field(default_factory=dict)
    thorax: str = "thoracic12"
    mirror_negate: tuple = ("pelvis_rotation", "pelvis_list", "pelvis_tz")
    viewer_port: int = 8080
    # The wrists are not in the state prior (SIPP's hand is rigid), and with the toolbox's 1 N m/rad spring the muscles that
    # cross elbow and wrist fold the hand by 20-40 deg as a by-product (active torque 0.6-2.3 N m in the first solutions).
    # A stiff passive wrist stands in for SIPP's rigid hand: 20 N m/rad keeps that by-product to a few degrees.
    passive_overrides: dict = field(default_factory=lambda: {
        j: {"spring_stiffness": 20.0} for j in ("flexion_r", "flexion_l", "deviation_r", "deviation_l")})

    @property
    def out(self):
        """Folder of the biosym model files."""
        return PROJECT / "models" / self.name

    @property
    def names_csv(self):
        """Muscle names with a uniform ``_r``/``_l`` suffix (``pair_muscles.py``)."""
        return self.out / "muscle_names.csv"


MODELS = {
    "ms_human_700": ModelConfig(
        name="ms_human_700", title="MS-Human-700 (full model, MuJoCo Menagerie)",
        mjcf=PROJECT / "models" / "original" / "ms_human_700" / "MS-Human-700.xml",
        # MuJoCo's own tendon lengths jump here (a wrapping side switch: TR2-TR5 at L5_S1_FE = -10 to -28 deg, by up to
        # 128 mm; PECM1 at shoulder_elv = 167 deg; DELT2 at shoulder_rot = 84 deg), and the shoulder's full range is more
        # than a polynomial of the walking envelope needs; the collocation bounds of these coordinates are this envelope
        fit_ranges_deg={"L5_S1_FE": (-8, 29), "elv_angle_r": (-45, 100), "elv_angle_l": (-45, 100),
                        "shoulder_elv_r": (0, 90), "shoulder_elv_l": (0, 90), "shoulder_rot_r": (-46, 75),
                        "shoulder_rot_l": (-46, 75)},
        thorax="thoracic12",
        mirror_negate=("pelvis_rotation", "pelvis_list", "pelvis_tz", "L5_S1_LB", "L5_S1_AR", "T12_L1_LB", "T12_L1_AR",
                       "T1_head_neck_LB", "T1_head_neck_AR")),
    "myofullbody": ModelConfig(
        name="myofullbody", title="MyoFullBody (MyoHub myo_sim)",
        mjcf=PROJECT / "models" / "original" / "myo_sim" / "myo_sim" / "models" / "myofullbody.xml",
        # no tendon-length jumps in the spine or shoulder (only RI2, 6-7 mm, in the index finger); the envelope keeps
        # the polynomials on the range walking needs: lumbar flexion within +-30 deg instead of -77..43, shoulder as
        # for MS-Human. The collocation bounds of these coordinates are this envelope (configs/myofullbody_*.yaml)
        fit_ranges_deg={"flex_extension": (-30, 30), "elv_angle_r": (-45, 100), "elv_angle_l": (-45, 100),
                        "shoulder_elv_r": (0, 90), "shoulder_elv_l": (0, 90), "shoulder_rot_r": (-46, 75),
                        "shoulder_rot_l": (-46, 75)},
        thorax="lumbar1",            # the torso, head and arm roots are rigid with lumbar1 and merged into it
        mirror_negate=("pelvis_rotation", "pelvis_list", "pelvis_tz", "lat_bending", "axial_rotation"),
        viewer_port=8082),
}


def from_argv(argv=None):
    """
    The `ModelConfig` named by the first command-line argument.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; ``sys.argv[1:]`` when omitted.

    Returns
    -------
        The configuration; exits with the list of models when the name is missing or unknown.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in MODELS:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} MODEL   (MODEL one of {', '.join(MODELS)})")
    return MODELS[argv[0]]
