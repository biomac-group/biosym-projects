"""
Build the biosym bundle of BioMAC's SIPP runmad model: models/sipp_runmad/.

Source: ``sipp_generic_runmad_smoothsphere.osim`` (copy in models/original/, SHA-256
2e296266...a3254, from the workstation's BiomechPriorVAE ``data/model``). That file
keeps its foot contact only as twelve ``CP_*`` markers (the marker is the centre of a
contact sphere) and has an empty ``ContactGeometrySet``; BioMAC's Gait3d class turns
the markers into spheres internally. The bundle is a plain OpenSim model:

* ``sipp_runmad.osim``: twelve ``ContactSphere`` (radius 0.032 m, centred on the old
  marker positions), a ``ContactHalfSpace`` floor (y = 0, normal +y) and twelve
  ``SmoothSphereHalfSpaceForce`` with the constants of BioMAC's
  ``contact/generate_contact_3d.py`` (E* 1e6 N/m^2, dissipation 2.0 s/m, static =
  dynamic friction 0.8, viscous friction 0.5, transition velocity 0.2 m/s) and no
  other properties, i.e. OpenSim's standard smoothing; ten arm ``CoordinateActuator``
  with gear ratio 10 N m (BioMAC's mExtraScaleFactor) and controls in [-5, 5] as in BioMAC's
  Gait3d (so up to +-50 N m). biosym keeps the control (state, bounded to [-5, 5]) and the gain (``optimal_force`` 10)
  apart, so the effort term penalises the control like BioMAC's effortTermTorques. The
  ``CP_*`` markers are dropped. The file loads in OpenSim.
* ``Geometry/``: the ``.vtp`` meshes the model references, next to the .osim as
  OpenSim (and the biosym viewer) look for them.

The model definition (``sipp_runmad.yaml``) and the contact (``contact_implicit.xml``) are maintained by hand and are not
written by this script.

Run from this directory:
    python scripts/make_sipp_model.py [--geometry-source DIR]
"""

import argparse
import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("models/original/sipp_generic_runmad_smoothsphere.osim")
OUT = Path("models/sipp_runmad")
EXPECTED_SHA = "2e296266317c52a8c666f36d0970ed86e6bdb4f9984e4af0717764d2f41a3254"
DEFAULT_GEOMETRY = Path.home() / "PhD/01_Python_Projects/biomechpriorVAE/plot/skeleton_frames/model/Geometry"

RADIUS = 0.032
HC = dict(stiffness=1.0e6, dissipation=2.0, static_friction=0.8, dynamic_friction=0.8,
          viscous_friction=0.5, transition_velocity=0.2)
ARMS = [f"{n}_{s}" for s in "rl" for n in ("arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup")]
GEAR = 10.0            # BioMAC Gait3d: mExtraScaleFactor = 10 N m
CONTROL_LIMIT = 5.0    # BioMAC Gait3d.update_controls: torque controls in [-5, 5], i.e. +-50 N m


def contact_block(markers):
    lines = ['<ContactGeometrySet name="contactgeometryset">', "\t<objects>",
             '\t\t<ContactHalfSpace name="floor">', "\t\t\t<socket_frame>/ground</socket_frame>",
             "\t\t\t<location>0 0 0</location>", "\t\t\t<orientation>0 0 -1.5707963267949</orientation>",
             "\t\t</ContactHalfSpace>"]
    for name, body, loc in markers:
        lines += [f'\t\t<ContactSphere name="sphere_{name}">', f"\t\t\t<socket_frame>{body}</socket_frame>",
                  f"\t\t\t<location>{loc}</location>", f"\t\t\t<radius>{RADIUS}</radius>", "\t\t</ContactSphere>"]
    lines += ["\t</objects>", "\t<groups/>", "</ContactGeometrySet>"]
    return "\n".join("\t\t" + l for l in lines)


def forces_block(markers):
    out = []
    for name, _, _ in markers:
        out.append(f'''				<SmoothSphereHalfSpaceForce name="contact_{name}">
					<socket_sphere>/contactgeometryset/sphere_{name}</socket_sphere>
					<socket_half_space>/contactgeometryset/floor</socket_half_space>
					<stiffness>{HC["stiffness"]:g}</stiffness>
					<dissipation>{HC["dissipation"]:g}</dissipation>
					<static_friction>{HC["static_friction"]:g}</static_friction>
					<dynamic_friction>{HC["dynamic_friction"]:g}</dynamic_friction>
					<viscous_friction>{HC["viscous_friction"]:g}</viscous_friction>
					<transition_velocity>{HC["transition_velocity"]:g}</transition_velocity>
				</SmoothSphereHalfSpaceForce>''')
    for c in ARMS:
        out.append(f'''				<CoordinateActuator name="{c}_act">
					<coordinate>{c}</coordinate>
					<optimal_force>{GEAR:g}</optimal_force>
					<min_control>-{CONTROL_LIMIT:g}</min_control>
					<max_control>{CONTROL_LIMIT:g}</max_control>
				</CoordinateActuator>''')
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-source", default=str(DEFAULT_GEOMETRY))
    args = ap.parse_args()

    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise SystemExit(f"{SRC} differs from the workstation original ({sha}).")
    text = SRC.read_text()
    root = ET.fromstring(text)
    markers = [(m.get("name")[3:], m.findtext("socket_parent_frame").strip(), " ".join(m.findtext("location").split()))
               for m in root.iter("Marker") if m.get("name", "").startswith("CP_")]
    if len(markers) != 12:
        raise SystemExit(f"expected 12 CP_ markers, found {len(markers)}")

    empty = re.search(r'[ \t]*<ContactGeometrySet name="contactgeometryset">\s*<objects/>\s*<groups/>\s*</ContactGeometrySet>', text)
    if empty is None:
        raise SystemExit("ContactGeometrySet is not the empty block this script expects")
    text = text[: empty.start()] + contact_block(markers) + text[empty.end():]

    forceset = text.index('<ForceSet name="forceset">')
    forceset_end = text.index("</ForceSet>", forceset)
    close = text.rindex("</objects>", forceset, forceset_end)      # the ForceSet's own, muscle paths have inner ones
    line_start = text.rfind("\n", 0, close) + 1
    text = text[:line_start] + forces_block(markers) + text[line_start:]

    text, dropped = re.subn(r'[ \t]*<Marker name="CP_[^"]*">.*?</Marker>\n?', "", text, flags=re.S)
    assert dropped == 12, dropped

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sipp_runmad.osim").write_text(text)

    meshes = sorted(set(re.findall(r"<mesh_file>([^<]+)</mesh_file>", text)))
    source = Path(args.geometry_source)
    missing = [m for m in meshes if not (source / m).exists()]
    if missing:
        raise SystemExit(f"geometry missing in {source}: {missing}")
    (OUT / "Geometry").mkdir(exist_ok=True)
    for m in meshes:
        shutil.copy2(source / m, OUT / "Geometry" / m)
    print(f"wrote {OUT}: osim (12 spheres, floor, 12 contact forces, {len(ARMS)} arm actuators, "
          f"{dropped} CP_ markers dropped), {len(meshes)} meshes, yaml and BioMAC-smoothing override")


if __name__ == "__main__":
    main()
