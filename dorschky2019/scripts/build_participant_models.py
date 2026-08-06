"""
Generate one scaled model file per participant.

Reads data/ParticipantInfo.csv and, for each subject, writes a height/mass-scaled
copy of models/gait2d.xml (body mass, inertia, and segment/COM offsets) plus a
matching Hunt-Crossley ground-contact file (contact point positions scaled the
same way) into models/participants/. A thin per-participant YAML then ties the
scaled body file to the scaled contact file and the shared (unscaled) actuator
file, using the Hunt-Crossley contact model everywhere per project convention.

Scaling law (isometric/geometric similarity, relative to the template's
reference 1.80 m / 75.0 kg):
    s_L = height / 1.80        (lengths: COM offsets, joint offsets, contact points)
    s_M = mass / 75.0          (masses)
    s_I = s_M * s_L**2         (inertias, mass * length^2)

Muscle/actuator geometry (optimal fiber length, tendon slack length, moment
arms, etc.) is intentionally left unscaled, matching the original scaling
scope used before this refactor.

Usage:
    python scripts/build_participant_models.py
"""
import os
import xml.etree.ElementTree as ET

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PARTICIPANTS_DIR = os.path.join(MODELS_DIR, "participants")

REF_HEIGHT_M = 1.80
REF_MASS_KG = 75.0


def _scale_vec3(s: str, factor: float) -> str:
    return " ".join(f"{float(v) * factor:.6g}" for v in s.split())


def build_scaled_body_xml(template_path: str, out_path: str, s_L: float, s_M: float, s_I: float) -> None:
    tree = ET.parse(template_path)
    root = tree.getroot()

    for site in root.iter("site"):
        if "pos" in site.attrib:
            site.set("pos", _scale_vec3(site.get("pos"), s_L))

    for body in root.iter("body"):
        if "pos" in body.attrib:
            body.set("pos", _scale_vec3(body.get("pos"), s_L))

    for inertial in root.iter("inertial"):
        inertial.set("pos", _scale_vec3(inertial.get("pos"), s_L))
        inertial.set("mass", f"{float(inertial.get('mass')) * s_M:.6g}")
        inertial.set("fullinertia", _scale_vec3(inertial.get("fullinertia"), s_I))

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def build_scaled_contact_xml(template_path: str, out_path: str, s_L: float) -> None:
    tree = ET.parse(template_path)
    root = tree.getroot()

    for cp in root.iter("contact_point"):
        if "pos" in cp.attrib:
            cp.set("pos", _scale_vec3(cp.get("pos"), s_L))

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def write_participant_yaml(out_path: str, subject_xml_name: str, contact_xml_name: str) -> None:
    content = f"""model:
  name: {subject_xml_name}
  additional_parameters:
    ground_contact:
      replace_existing: true
      file: "{contact_xml_name}" # Hunt-Crossley, project-wide default (see models/gait2d_huntcrossley.yaml)
    actuators:
      replace_existing: true
      file: "../gait2d_actuators.xml" # shared, unscaled muscle geometry
"""
    with open(out_path, "w") as f:
        f.write(content)


def main():
    os.makedirs(PARTICIPANTS_DIR, exist_ok=True)
    csv_path = os.path.join(PROJECT_DIR, "data", "ParticipantInfo.csv")
    df_info = pd.read_csv(csv_path)

    body_template = os.path.join(MODELS_DIR, "gait2d.xml")
    contact_template = os.path.join(MODELS_DIR, "gait2d_ground_contact_huntcrossley.xml")

    for _, row in df_info.iterrows():
        subject_id = row["ID"]
        height = float(row["BODYHEIGHT"])
        mass = float(row["BODYMASS"])
        s_L = height / REF_HEIGHT_M
        s_M = mass / REF_MASS_KG
        s_I = s_M * (s_L ** 2)

        body_xml_name = f"{subject_id}.xml"
        contact_xml_name = f"{subject_id}_ground_contact_huntcrossley.xml"
        yaml_name = f"{subject_id}.yaml"

        build_scaled_body_xml(
            body_template, os.path.join(PARTICIPANTS_DIR, body_xml_name), s_L, s_M, s_I
        )
        build_scaled_contact_xml(
            contact_template, os.path.join(PARTICIPANTS_DIR, contact_xml_name), s_L
        )
        write_participant_yaml(
            os.path.join(PARTICIPANTS_DIR, yaml_name), body_xml_name, contact_xml_name
        )
        print(f"Wrote participant model for {subject_id} (height={height:.2f}m, mass={mass:.1f}kg, "
              f"s_L={s_L:.4f}, s_M={s_M:.4f}, s_I={s_I:.4f})")


if __name__ == "__main__":
    main()
