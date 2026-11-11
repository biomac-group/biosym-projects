"""
Turn a regenerated ``sipp_generic_runmad.osim`` (mrigait_empkins/sipp_runmad/create_runmad_sipp.py)
into the biosym SIPP runmad bundle model ``models/sipp_runmad/sipp_runmad.osim``.

Applies the same post-processing as ``make_sipp_model.py`` and changes nothing else in the regenerated file:

* the 12 SIPP contact spheres (positions read from the ``CP_*`` markers of the BioMAC smoothsphere original,
  models/original/sipp_generic_runmad_smoothsphere.osim, the same positions as in contact_*.xml), the floor
  half space and 12 SmoothSphereHalfSpaceForce are inserted (the yamls replace the contact by an XML anyway);
* the ten arm CoordinateActuator (gear 10 N m, controls in [-5, 5]) are appended to the ForceSet;
* the 16 ``CP_*`` box-corner markers of the regenerated file are dropped, as ``make_sipp_model.py`` does with the
  smoothsphere markers.

Run from the 3d_simulations directory:
    python scripts/regen_sipp_model.py SOURCE.osim OUTPUT.osim
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_sipp_model import contact_block, forces_block  # noqa: E402

SPHERE_SRC = Path("models/original/sipp_generic_runmad_smoothsphere.osim")


def main(src, out):
    """
    Post-process a regenerated SIPP generic model into the biosym bundle model.

    Parameters
    ----------
    src : str
        Path of the regenerated ``sipp_generic_runmad.osim``.
    out : str
        Path the post-processed model is written to.
    """
    root = ET.parse(SPHERE_SRC).getroot()
    markers = [(m.get("name")[3:], m.findtext("socket_parent_frame").strip(), " ".join(m.findtext("location").split()))
               for m in root.iter("Marker") if m.get("name", "").startswith("CP_")]
    if len(markers) != 12:
        raise SystemExit(f"expected 12 CP_ markers in {SPHERE_SRC}, found {len(markers)}")

    text = Path(src).read_text()
    empty = re.search(r'[ \t]*<ContactGeometrySet name="contactgeometryset">\s*<objects\s*/>\s*<groups\s*/>\s*</ContactGeometrySet>', text)
    if empty is None:
        raise SystemExit("ContactGeometrySet is not the empty block this script expects")
    text = text[: empty.start()] + contact_block(markers) + text[empty.end():]

    forceset = text.index('<ForceSet name="forceset">')
    forceset_end = text.index("</ForceSet>", forceset)
    close = text.rindex("</objects>", forceset, forceset_end)
    line_start = text.rfind("\n", 0, close) + 1
    text = text[:line_start] + forces_block(markers) + text[line_start:]

    text, dropped = re.subn(r'[ \t]*<Marker name="CP_[^"]*">.*?</Marker>\n?', "", text, flags=re.S)
    print(f"dropped {dropped} CP_ markers")
    Path(out).write_text(text)
    ET.parse(out)  # well-formed


if __name__ == "__main__":
    main(*sys.argv[1:3])
