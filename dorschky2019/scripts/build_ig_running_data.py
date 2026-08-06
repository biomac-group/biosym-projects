"""
Generate a population-average running reference (joint angles + GRF) used as
the tracking target for the ig_running warm-start stage, mirroring the role
gait_avg_joint_angles.mot/gait_avg_grf.mot play for ig_walking.

Source: data/fukuchi_running_raw/RBDS0{01..28}runT35.mat -- one file per
subject, 3.5 m/s treadmill running trial, from Fukuchi et al. (PeerJ, 2017),
via BioMAC-Sim-Toolbox's TrackingData format. Each file holds only
'hip'/'knee'/'Fx'/'Fy' mean arrays (100 samples, one full gait cycle for one
leg): the original per-subject MATLAB TrackingData objects also expose
pelvis/ankle/GRF-z channels, but extracting those requires re-parsing an
MCOS-serialized MATLAB table, which needs a live MATLAB license (unavailable
here) -- see convert_fukuchi_to_plain.m in the biomechpriorVAE project for
the extraction that produced these per-subject files. Pelvis translation/tilt
has no data and is excluded from tracking in ig_running.yaml
(build_configs.py). Ankle angle also has no data (tracked to a zero-mean
placeholder here) but is intentionally not excluded.

Because each subject's series only covers one leg's full cycle (nonzero GRF
during that leg's own stance, ~0-48% of the cycle), the opposite leg is
synthesized by a 50%-phase circular shift, assuming left/right symmetry --
the same assumption BioMAC-Sim-Toolbox's TrackingData.isSymmetric encodes.

GRF channels are normalized to bodyweight multiples (dimensionless) in the
source data; they are converted to Newtons using the mean mass of the 28
Fukuchi subjects (not per-participant mass), since this file -- like
gait_avg_grf.mot -- is a population average, not subject-specific.

Variance is the cross-subject population variance (28 subjects). GRF
variance is floored at MIN_GRF_VARIANCE (Newtons^2) so that samples with
near-zero cross-subject spread (e.g. the swing-phase zero-force plateau)
don't blow up the weighted tracking error; joint-angle variance and the
no-data placeholder columns are not floored.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

from biosym.utils import write_mot

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "fukuchi_running_raw")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

N_POINTS = 100
GRAVITY = 9.81
MEAN_SUBJECT_MASS = 69.6375  # kg, mean of the 28 RBDS subjects at 3.5 m/s
MIN_GRF_VARIANCE = 1.0  # N^2 floor -- cross-subject GRF variance alone was too low

# The raw Fukuchi cycle is anchored at right-foot touchdown (t=0). Individual
# subject running trials in this project (e.g. P0x_normrunning_mean_FP.mat.parquet)
# are anchored differently -- their own GRF_y_r stays at zero until ~15% into
# the cycle. Cross-correlating fukuchi_running_grf.mot's right-foot GRF against
# P05_normrunning's real GRF_y_r (before this shift) gave a near-perfect match
# (corr=0.998) at a +14-sample circular roll; rounded to 15 here. Rolling the
# raw per-subject arrays by this amount (before the existing left/right 50%
# split below) re-anchors the whole reference cycle to match that convention.
CYCLE_SHIFT = 15

# Column order must match the model's coordinate order -- mirrors
# segment_gait_cycles.create_averaged_gait_joint_angles's column_order,
# already validated against this project's gait2d model via ig_walking.
JOINT_ANGLE_COLUMNS = [
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tilt",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
]

# No real data available for these -- filled with a placeholder (zero mean,
# unit variance). Pelvis columns are excluded from tracking in
# ig_running.yaml; ankle columns are not (tracked toward the zero placeholder).
NO_DATA_COLUMNS = {"pelvis_tx", "pelvis_ty", "pelvis_tilt", "ankle_angle_r", "ankle_angle_l"}


def load_subject(path: str) -> dict:
    d = loadmat(path, simplify_cells=True)
    roll = lambda a: np.roll(np.asarray(a, dtype=float), CYCLE_SHIFT)
    return {
        "hip": roll(d["hip"]["mean"]),
        "knee": roll(d["knee"]["mean"]),
        "Fx": roll(d["Fx"]["mean"]),
        "Fy": roll(d["Fy"]["mean"]),
    }


def build_joint_angles(subjects: list[dict]) -> pd.DataFrame:
    hip = np.stack([s["hip"] for s in subjects])  # (n_subjects, 100), right leg
    knee = np.stack([s["knee"] for s in subjects])

    shift = N_POINTS // 2
    hip_l = np.roll(hip, shift, axis=1)
    knee_l = np.roll(knee, shift, axis=1)

    data = {}
    for col in JOINT_ANGLE_COLUMNS:
        if col in NO_DATA_COLUMNS:
            data[f"{col}_mean"] = np.zeros(N_POINTS)
            data[f"{col}_var"] = np.ones(N_POINTS)
        elif col == "hip_flexion_r":
            data[f"{col}_mean"] = hip.mean(axis=0)
            data[f"{col}_var"] = hip.var(axis=0)
        elif col == "hip_flexion_l":
            data[f"{col}_mean"] = hip_l.mean(axis=0)
            data[f"{col}_var"] = hip_l.var(axis=0)
        elif col == "knee_angle_r":
            data[f"{col}_mean"] = knee.mean(axis=0)
            data[f"{col}_var"] = knee.var(axis=0)
        elif col == "knee_angle_l":
            data[f"{col}_mean"] = knee_l.mean(axis=0)
            data[f"{col}_var"] = knee_l.var(axis=0)

    ordered_cols = [c for col in JOINT_ANGLE_COLUMNS for c in (f"{col}_mean", f"{col}_var")]
    return pd.DataFrame(data)[ordered_cols]


def build_grf(subjects: list[dict]) -> pd.DataFrame:
    fx = np.stack([s["Fx"] for s in subjects]) * MEAN_SUBJECT_MASS * GRAVITY
    fy = np.stack([s["Fy"] for s in subjects]) * MEAN_SUBJECT_MASS * GRAVITY

    shift = N_POINTS // 2
    fx_l = np.roll(fx, shift, axis=1)
    fy_l = np.roll(fy, shift, axis=1)

    # Cross-subject variance floored at MIN_GRF_VARIANCE (see module docstring).
    fx_l_var = np.maximum(fx_l.var(axis=0), MIN_GRF_VARIANCE)
    fy_l_var = np.maximum(fy_l.var(axis=0), MIN_GRF_VARIANCE)
    fx_var = np.maximum(fx.var(axis=0), MIN_GRF_VARIANCE)
    fy_var = np.maximum(fy.var(axis=0), MIN_GRF_VARIANCE)
    # No mediolateral (z) data at all -- floor value stands in for its variance.
    fz_var = np.full(N_POINTS, MIN_GRF_VARIANCE)

    data = {
        "1_ground_force_vx_mean": fx_l.mean(axis=0),
        "1_ground_force_vx_var": fx_l_var,
        "1_ground_force_vy_mean": fy_l.mean(axis=0),
        "1_ground_force_vy_var": fy_l_var,
        "1_ground_force_vz_mean": np.zeros(N_POINTS),
        "1_ground_force_vz_var": fz_var,
        "ground_force_vx_mean": fx.mean(axis=0),
        "ground_force_vx_var": fx_var,
        "ground_force_vy_mean": fy.mean(axis=0),
        "ground_force_vy_var": fy_var,
        "ground_force_vz_mean": np.zeros(N_POINTS),
        "ground_force_vz_var": fz_var,
    }
    ordered_cols = [
        "1_ground_force_vx_mean", "1_ground_force_vx_var",
        "1_ground_force_vy_mean", "1_ground_force_vy_var",
        "1_ground_force_vz_mean", "1_ground_force_vz_var",
        "ground_force_vx_mean", "ground_force_vx_var",
        "ground_force_vy_mean", "ground_force_vy_var",
        "ground_force_vz_mean", "ground_force_vz_var",
    ]
    return pd.DataFrame(data)[ordered_cols]


def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*runT35.mat")))
    if not paths:
        raise FileNotFoundError(f"No Fukuchi 3.5 m/s files found in {RAW_DIR}")
    subjects = [load_subject(p) for p in paths]
    print(f"Loaded {len(subjects)} subjects from {RAW_DIR}")

    angles_df = build_joint_angles(subjects)
    grf_df = build_grf(subjects)

    write_mot(angles_df, os.path.join(DATA_DIR, "fukuchi_running_joint_angles.mot"), name="Coordinates")
    write_mot(grf_df, os.path.join(DATA_DIR, "fukuchi_running_grf.mot"), name="Averaged GRF")
    print("Wrote fukuchi_running_joint_angles.mot and fukuchi_running_grf.mot")


if __name__ == "__main__":
    main()
