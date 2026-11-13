"""
Fast-twitch fibre fraction of the muscles of this project's models, for the metabolic cost (``cot_term``).

Neither model has fibre types. The BioMAC toolbox's table (``biosym.model.actuators.metabolics.FAST_TWITCH``) gives them for
the muscles of its gait3d model; the muscles of MS-Human-700 and MyoFullBody that correspond to one of them (leg muscles by
name, the erector spinae and the oblique abdominal muscles by group) take its value, every other muscle (the rest of the
trunk, neck, arms, hands) 0.5. ``fast_twitch(name)`` is that map; ``make_model.py`` writes it as ``fasttwitch`` into the
muscle file, and

    uv run python scripts/fibre_types.py MODEL        # adds it to an existing muscle file without a refit
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import from_argv  # noqa: E402

DEFAULT = 0.5
# model muscle (without side, without the number of its parts) -> gait3d muscle of the toolbox table (keys of FAST_TWITCH)
CORRESPONDING = {
    "addbrev": "add_brev", "addlong": "add_long", "bflh": "bifemlh", "bfsh": "bifemsh", "edl": "ext_dig", "ehl": "ext_hal",
    "fdl": "flex_dig", "fhl": "flex_hal", "gaslat": "lat_gas", "gasmed": "med_gas", "grac": "grac", "iliacus": "iliacus",
    "perbrev": "per_brev", "perlong": "per_long", "piri": "peri", "psoas": "psoas", "recfem": "rect_fem", "sart": "sar",
    "semimem": "semimem", "semiten": "semiten", "soleus": "soleus", "tfl": "tfl", "tibant": "tib_ant", "tibpost": "tib_post",
    "vasint": "vas_int", "vaslat": "vas_lat", "vasmed": "vas_med", "pect": "pect",
}
# groups with a number of parts: model prefix (regex on the stem) -> table key
GROUPS = [(r"^addmag", "add_mag1"), (r"^glmax\d", "glut_max1"), (r"^glmed\d", "glut_med1"), (r"^glmin\d", "glut_min1"),
          (r"^(IL|LTpL|LTpT)(_|$)", "ercspn"), (r"^IO(_|$)", "intobl"), (r"^EO(_|$)", "extobl")]


def fast_twitch(name):
    """
    Fast-twitch fraction of one muscle.

    Parameters
    ----------
    name : str
        Muscle name with or without its ``_r``/``_l`` suffix.

    Returns
    -------
        The fraction of the corresponding gait3d muscle, else 0.5.
    """
    from biosym.model.actuators.metabolics import FAST_TWITCH

    stem = re.sub(r"_[rl]$", "", name)
    key = CORRESPONDING.get(stem) or next((k for pat, k in GROUPS if re.match(pat, stem)), None)
    return FAST_TWITCH[key] if key else DEFAULT


def main():
    cfg = from_argv()
    path = cfg.out / f"{cfg.name}_muscles.xml"
    tree = ET.parse(path)
    n_table = 0
    for el in tree.getroot().iter("mujoco_muscle"):
        value = fast_twitch(el.get("name"))
        n_table += value != DEFAULT
        el.set("fasttwitch", f"{value:.6g}")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"{path.name}: fasttwitch written, {n_table} muscles from the toolbox table, the rest {DEFAULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
