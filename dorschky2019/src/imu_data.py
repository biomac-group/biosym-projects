"""
Reusable loaders for the Dorschky et al. 2019 participant/trial data.

Keeps the parquet -> 42-channel IMU tracking array assembly, and participant
metadata (height/mass/IMU sensor offsets) parsing, in one place so
run_experiment.py can stay a thin CLI wrapper.
"""
import numpy as np
import pandas as pd


def load_subject_metadata(csv_path: str, subject_id: str):
    """
    Parse ParticipantInfo.csv for a subject's height, mass, and 3D IMU sensor
    offsets (local segment frame).

    Returns (height_m, mass_kg, sensor_config) where sensor_config is a list
    of {"body": ..., "offset": [x, y, z]} dicts, as expected by IMUSensorModel.
    """
    df_info = pd.read_csv(csv_path)
    row = df_info[df_info["ID"] == subject_id].iloc[0]
    height = float(row["BODYHEIGHT"])
    mass = float(row["BODYMASS"])

    sensor_config = [
        {"body": "pelvis", "offset": [float(row["IMU_PELVIS_PX"]), float(row["IMU_PELVIS_PY"]), float(row["IMU_PELVIS_PZ"])]},
        {"body": "femur_r", "offset": [float(row["IMU_FEMUR_R_PX"]), float(row["IMU_FEMUR_R_PY"]), float(row["IMU_FEMUR_R_PZ"])]},
        {"body": "tibia_r", "offset": [float(row["IMU_TIBIA_R_PX"]), float(row["IMU_TIBIA_R_PY"]), float(row["IMU_TIBIA_R_PZ"])]},
        {"body": "foot_r", "offset": [float(row["IMU_FOOT_R_PX"]), float(row["IMU_FOOT_R_PY"]), float(row["IMU_FOOT_R_PZ"])]},
        {"body": "femur_l", "offset": [float(row["IMU_FEMUR_L_PX"]), float(row["IMU_FEMUR_L_PY"]), float(row["IMU_FEMUR_L_PZ"])]},
        {"body": "tibia_l", "offset": [float(row["IMU_TIBIA_L_PX"]), float(row["IMU_TIBIA_L_PY"]), float(row["IMU_TIBIA_L_PZ"])]},
        {"body": "foot_l", "offset": [float(row["IMU_FOOT_L_PX"]), float(row["IMU_FOOT_L_PY"]), float(row["IMU_FOOT_L_PZ"])]},
    ]
    return height, mass, sensor_config


def _extract_sensor_rows(df: pd.DataFrame, type_name: str, segment_name: str):
    """Extract the 3 (X, Y, Z) rows for a segment's sensor. Returns (mean, var), each (3, 100)."""
    rows = df[(df["type"] == type_name) & (df["name"] == segment_name)]
    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows for type {type_name} and segment {segment_name}, found {len(rows)}")
    mean_arr = np.stack([np.array(r["mean"]) for _, r in rows.iterrows()])
    var_arr = np.stack([np.array(r["var"]) for _, r in rows.iterrows()])
    var_arr = np.maximum(var_arr, 1e-5)  # epsilon-guard against zero variance
    return mean_arr, var_arr


SEGMENTS = ["pelvis", "femur_r", "tibia_r", "foot_r", "femur_l", "tibia_l", "foot_l"]
LEFT_SEGMENTS = ["femur_l", "tibia_l", "foot_l"]


def load_imu_tracking_data(data_dir: str, subject_id: str, trial_type: str, nnodes: int):
    """
    Load a subject/trial's real IMU data and assemble it into the
    (nnodes + 1, 42) tracking arrays expected by IMUTrackingObjective, plus
    the experimental right-side joint angles (for validation plots only).

    Returns a dict with keys: dur_full, speed, t_col, imu_mean, imu_var,
    angles_exp (dict of hip/knee/ankle_r -> array on the same t_col grid).
    """
    import os

    parquet_path = os.path.join(data_dir, f"{subject_id}_{trial_type}_mean_FP.mat.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet tracking data file not found at: {parquet_path}")
    df_pq = pd.read_parquet(parquet_path)

    dur_full = float(df_pq[df_pq["type"] == "dur"].iloc[0]["mean"][0])
    speed = float(df_pq[df_pq["type"] == "speed"].iloc[0]["mean"][0])

    segment_data = {}
    for seg in SEGMENTS:
        acc_mean, acc_var = _extract_sensor_rows(df_pq, "acc", seg)
        gyr_mean, gyr_var = _extract_sensor_rows(df_pq, "gyro", seg)
        segment_data[seg] = {"acc_mean": acc_mean, "acc_var": acc_var, "gyr_mean": gyr_mean, "gyr_var": gyr_var}

    full_imu_mean = np.zeros((100, 42))
    full_imu_var = np.zeros((100, 42))

    def insert_sensor(arr, idx, acc, gyr):
        base = idx * 6
        arr[:, base:base + 3] = acc.T
        arr[:, base + 3:base + 6] = gyr.T

    for idx, seg in enumerate(SEGMENTS):
        insert_sensor(full_imu_mean, idx, segment_data[seg]["acc_mean"], segment_data[seg]["gyr_mean"])
        insert_sensor(full_imu_var, idx, segment_data[seg]["acc_var"], segment_data[seg]["gyr_var"])

    # Make periodic (101 points) and interpolate onto the collocation grid (nnodes + 1 points).
    full_imu_mean_101 = np.vstack([full_imu_mean, full_imu_mean[0:1]])
    full_imu_var_101 = np.vstack([full_imu_var, full_imu_var[0:1]])
    t_exp = np.linspace(0, dur_full, 101)
    t_col = np.linspace(0, dur_full, nnodes + 1)
    col_imu_mean = np.zeros((nnodes + 1, 42))
    col_imu_var = np.zeros((nnodes + 1, 42))
    for c in range(42):
        col_imu_mean[:, c] = np.interp(t_col, t_exp, full_imu_mean_101[:, c])
        col_imu_var[:, c] = np.interp(t_col, t_exp, full_imu_var_101[:, c])

    angles_exp = {}
    for name, col in (("hip_r", "hip_flexion_r"), ("knee_r", "knee_angle_r"), ("ankle_r", "ankle_angle_r")):
        exp = np.array(df_pq[df_pq["name"] == col].iloc[0]["mean"])
        exp_101 = np.append(exp, exp[0])
        angles_exp[name] = np.interp(t_col, t_exp, exp_101)

    return {
        "dur_full": dur_full,
        "speed": speed,
        "t_col": t_col,
        "imu_mean": col_imu_mean,
        "imu_var": col_imu_var,
        "angles_exp": angles_exp,
    }
