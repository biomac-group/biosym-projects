import os
import sys
import yaml
import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolve paths
scripts_dir = os.path.dirname(os.path.abspath(__file__))
demo_dir = os.path.dirname(scripts_dir)
repo_root = os.path.dirname(demo_dir)
src_dir = os.path.join(demo_dir, "src")

for p in (src_dir, demo_dir, repo_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from biosym.ocp import collocation
from biosym.model.model import load_model
import biosym.ocp.objfun as boo
try:
    import track_imu
except ImportError:
    from src import track_imu
try:
    from sensors import IMUSensorModel
except ImportError:
    from src.sensors import IMUSensorModel

def load_subject_metadata(csv_path, subject_id="P02"):
    """
    Parse ParticipantInfo.csv to extract subject physical parameters and 3D IMU offsets.
    """
    print(f"Loading participant metadata from {csv_path}")
    df_info = pd.read_csv(csv_path)
    row = df_info[df_info['ID'] == subject_id].iloc[0]
    height = float(row['BODYHEIGHT'])
    mass = float(row['BODYMASS'])
    
    # 3D sensor offsets in local segment frame
    sensor_config = [
        {"body": "pelvis", "offset": [float(row["IMU_PELVIS_PX"]), float(row["IMU_PELVIS_PY"]), float(row["IMU_PELVIS_PZ"])]},
        {"body": "femur_r", "offset": [float(row["IMU_FEMUR_R_PX"]), float(row["IMU_FEMUR_R_PY"]), float(row["IMU_FEMUR_R_PZ"])]},
        {"body": "tibia_r", "offset": [float(row["IMU_TIBIA_R_PX"]), float(row["IMU_TIBIA_R_PY"]), float(row["IMU_TIBIA_R_PZ"])]},
        {"body": "foot_r", "offset": [float(row["IMU_FOOT_R_PX"]), float(row["IMU_FOOT_R_PY"]), float(row["IMU_FOOT_R_PZ"])]},
        {"body": "femur_l", "offset": [float(row["IMU_FEMUR_L_PX"]), float(row["IMU_FEMUR_L_PY"]), float(row["IMU_FEMUR_L_PZ"])]},
        {"body": "tibia_l", "offset": [float(row["IMU_TIBIA_L_PX"]), float(row["IMU_TIBIA_L_PY"]), float(row["IMU_TIBIA_L_PZ"])]},
        {"body": "foot_l", "offset": [float(row["IMU_FOOT_L_PX"]), float(row["IMU_FOOT_L_PY"]), float(row["IMU_FOOT_L_PZ"])]},
    ]
    print(f"  Subject ID: {subject_id}")
    print(f"  Height: {height:.2f} m | Mass: {mass:.1f} kg")
    return height, mass, sensor_config

def extract_sensor_data(df, type_name, segment_name):
    """
    Extract the 3 rows corresponding to X, Y, Z directions for a segment's sensor.
    Returns:
        mean_arr: np.ndarray of shape (3, 100)
        var_arr: np.ndarray of shape (3, 100), epsilon-guarded to be >= 1e-5
    """
    rows = df[(df['type'] == type_name) & (df['name'] == segment_name)]
    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows for type {type_name} and segment {segment_name}, found {len(rows)}")
    mean_arr = np.stack([np.array(r['mean']) for _, r in rows.iterrows()])
    var_arr = np.stack([np.array(r['var']) for _, r in rows.iterrows()])
    
    # Epsilon-guard the variance directly during parsing/extraction as requested
    var_arr = np.maximum(var_arr, 1e-5)
    
    return mean_arr, var_arr

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Muscle-Driven IMU-Only Gait Tracking of Real Participant Data using BioSYM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--subject", type=str, default="P02", choices=[f"P{i:02d}" for i in range(1, 11)],
                        help="Subject ID to track (P01 to P10)")
    parser.add_argument("--trial-type", type=str, default="normwalking",
                        choices=["slowwalking", "normwalking", "fastwalking", "slowrunning", "normrunning", "fastrunning"],
                        help="Gait trial condition")
    parser.add_argument("--nodes", type=int, default=100,
                        help="Number of collocation nodes for the gait cycle (collocation grid evaluates at nodes+1 points)")
    parser.add_argument("--max-iter", type=int, default=10000,
                        help="Maximum CyIPOPT solver iterations")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="IPOPT convergence tolerance")
    parser.add_argument("--weight-imu", type=float, default=10,
                        help="Objective weight for IMU signal tracking")
    parser.add_argument("--weight-effort", type=float, default=1,
                        help="Objective weight for muscle effort (activation cubed)")
    parser.add_argument("--initial-guess", type=str, default=os.path.join(demo_dir, "result", "walking2d.pkl"),
                        help="Path to pre-solved initial guess walking trajectory")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output path to save the tracked collocation solution (.pkl). Defaults to dorschky_demo/result/tracked_{subject}_{trial}.pkl")
    parser.add_argument("--plot-file", type=str, default=None,
                        help="Path to save verification plot. Defaults to dorschky_demo/{subject}_{trial}_results.png")
    parser.add_argument("--no-solve", action="store_true",
                        help="Set up OCP but do not run the IPOPT solver (useful for quick configuration checks)")
    parser.add_argument("--launch-dashboard", action="store_true",
                        help="Launch the interactive OCP dashboard during the optimal control solve")
    
    args = parser.parse_args()

    subject_id = args.subject
    trial_type = args.trial_type
    nnodes = args.nodes
    
    print(f"=== Muscle-Driven IMU-Only Gait Tracking of Real Participant Data ===")
    print(f"  Target: Subject {subject_id} | Trial: {trial_type}")
    
    # 1. Register custom IMU Tracking Objective programmatically
    print("Injecting custom IMUTrackingObjective into biosym namespaces...")
    boo.IMUTrackingObjective = track_imu
    
    # Load participant metadata and config
    data_dir = os.path.join(demo_dir, "data")
    csv_path = os.path.join(data_dir, "ParticipantInfo.csv")
    height, mass, sensor_config = load_subject_metadata(csv_path, subject_id=subject_id)
    
    # 2. Process real IMU tracking data from parquet
    parquet_path = os.path.join(data_dir, f"{subject_id}_{trial_type}_mean_FP.mat.parquet")
    print(f"Loading parquet tracking data from {parquet_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet tracking data file not found at: {parquet_path}")
        
    df_pq = pd.read_parquet(parquet_path)
    
    # Extract dur and speed
    dur_full = df_pq[df_pq['type'] == 'dur'].iloc[0]['mean'][0]
    speed = df_pq[df_pq['type'] == 'speed'].iloc[0]['mean'][0]
    dur_half = dur_full / 2.0
    
    print(f"  Gait Cycle Duration: {dur_full:.4f} s (Half Cycle: {dur_half:.4f} s)")
    print(f"  Mean Speed: {speed:.4f} m/s")
    
    # Extract raw 100-node experimental data and variance (without averaging limbs)
    print("Extracting raw independent segment proper accelerations and angular velocities...")
    pelvis_acc_mean, pelvis_acc_var = extract_sensor_data(df_pq, 'acc', 'pelvis')
    pelvis_gyr_mean, pelvis_gyr_var = extract_sensor_data(df_pq, 'gyro', 'pelvis')
    
    segments = ['femur_r', 'tibia_r', 'foot_r', 'femur_l', 'tibia_l', 'foot_l']
    limbs_data = {}
    for seg in segments:
        acc_mean, acc_var = extract_sensor_data(df_pq, 'acc', seg)
        gyr_mean, gyr_var = extract_sensor_data(df_pq, 'gyro', seg)
        limbs_data[seg] = {
            'acc_mean': acc_mean,
            'acc_var': acc_var,
            'gyr_mean': gyr_mean,
            'gyr_var': gyr_var
        }
        
    # Roll left-limb signals by 50 nodes along axis 1 (100 timepoints) to align them with right heel strike
    left_segments = ['femur_l', 'tibia_l', 'foot_l']
    for seg in left_segments:
        limbs_data[seg]['acc_mean'] = np.roll(limbs_data[seg]['acc_mean'], 50, axis=1)
        limbs_data[seg]['acc_var'] = np.roll(limbs_data[seg]['acc_var'], 50, axis=1)
        limbs_data[seg]['gyr_mean'] = np.roll(limbs_data[seg]['gyr_mean'], 50, axis=1)
        limbs_data[seg]['gyr_var'] = np.roll(limbs_data[seg]['gyr_var'], 50, axis=1)

        
    # Assemble full 42-channel array of shape (100, 42) for mean and variance
    full_imu_mean = np.zeros((100, 42))
    full_imu_var = np.zeros((100, 42))
    
    def insert_sensor(arr, idx, acc, gyr):
        base = idx * 6
        arr[:, base:base+3] = acc.T
        arr[:, base+3:base+6] = gyr.T
        
    insert_sensor(full_imu_mean, 0, pelvis_acc_mean, pelvis_gyr_mean)
    insert_sensor(full_imu_var, 0, pelvis_acc_var, pelvis_gyr_var)
    
    for idx, seg in enumerate(segments, start=1):
        insert_sensor(full_imu_mean, idx, limbs_data[seg]['acc_mean'], limbs_data[seg]['gyr_mean'])
        insert_sensor(full_imu_var, idx, limbs_data[seg]['acc_var'], limbs_data[seg]['gyr_var'])
    
    # Make arrays periodic (101 points) by appending the first sample (index 0) to the end (index 100)
    full_imu_mean_101 = np.vstack([full_imu_mean, full_imu_mean[0:1]])
    full_imu_var_101 = np.vstack([full_imu_var, full_imu_var[0:1]])
    
    # Interpolate onto collocation nodes covering the full cycle (since there are nnodes + 1 grid points)
    print(f"Interpolating periodic IMU tracking data from 101 to {nnodes + 1} nodes...")
    col_imu_mean = np.zeros((nnodes + 1, 42))
    col_imu_var = np.zeros((nnodes + 1, 42))
    t_exp = np.linspace(0, dur_full, 101)
    t_col = np.linspace(0, dur_full, nnodes + 1)
    for c in range(42):
        col_imu_mean[:, c] = np.interp(t_col, t_exp, full_imu_mean_101[:, c])
        col_imu_var[:, c] = np.interp(t_col, t_exp, full_imu_var_101[:, c])
        
    # Also extract right experimental joint angles to validate kinematic reconstruction
    q_hip_r_exp = np.array(df_pq[df_pq['name'] == 'hip_flexion_r'].iloc[0]['mean'])
    q_knee_r_exp = np.array(df_pq[df_pq['name'] == 'knee_angle_r'].iloc[0]['mean'])
    q_ankle_r_exp = np.array(df_pq[df_pq['name'] == 'ankle_angle_r'].iloc[0]['mean'])
    
    # Make joint angles periodic by appending the first sample to the end
    q_hip_r_exp_101 = np.append(q_hip_r_exp, q_hip_r_exp[0])
    q_knee_r_exp_101 = np.append(q_knee_r_exp, q_knee_r_exp[0])
    q_ankle_r_exp_101 = np.append(q_ankle_r_exp, q_ankle_r_exp[0])
    
    q_hip_r_exp_col = np.interp(t_col, t_exp, q_hip_r_exp_101)
    q_knee_r_exp_col = np.interp(t_col, t_exp, q_knee_r_exp_101)
    q_ankle_r_exp_col = np.interp(t_col, t_exp, q_ankle_r_exp_101)
    
    # 3. Model parameter scaling and setup
    model_yaml = os.path.join(demo_dir, "models/gait2d.yaml")
    print(f"Loading core muscle-driven model from {model_yaml}")
    model = load_model(model_yaml)
    
    # Scale model constants relative to standard height/mass (1.80m / 75.0kg)
    s_L = height / 1.80
    s_M = mass / 75.0
    s_I = s_M * (s_L ** 2)
    
    print("Scaling physical constants:")
    print(f"  Length scale (COM/Joint offsets): {s_L:.4f}")
    print(f"  Mass scale (Segment masses): {s_M:.4f}")
    print(f"  Inertia scale (Segment inertias): {s_I:.4f}")
    
    model.default_constants = model.default_constants.replace(
        mass=model.default_constants.mass * s_M,
        inertia=model.default_constants.inertia * s_I,
        com=model.default_constants.com * s_L,
        offset=model.default_constants.offset * s_L
    )
    
    # Load standard periodic walking config
    walking_yaml = os.path.join(demo_dir, "configs", "track_imu2d.yaml")
    print(f"Loading standard walking OCP configuration from {walking_yaml}")
    with open(walking_yaml, "r") as f:
        config = yaml.safe_load(f)["collocation"]
        
    if "initial_guess" in config and config["initial_guess"].get("type") == "from_file":
        config["initial_guess"]["file"] = os.path.abspath(
            os.path.join(demo_dir, "configs", config["initial_guess"]["file"])
        )

    # Configure settings for tracking
    config["settings"]["nnodes"] = nnodes
    
    if args.output_file is None:
        output_file = os.path.join(demo_dir, "result", f"tracked_{subject_id}_{trial_type}.pkl")
    else:
        output_file = args.output_file
    config["settings"]["output"] = {"file": output_file}
    
    config["settings"]["max_iter"] = args.max_iter
    config["settings"]["tol"] = args.tol
    
    # Match subject duration and speed (full cycle)
    config["bounds"]["dur"] = [float(dur_full), float(dur_full)]
    
    # Set objectives for tracking: strictly IMU + effort
    config["objectives"] = [
        {
            "name": "IMUTrackingObjective",
            "weight": args.weight_imu,
            "args": {
                "tracking_data": col_imu_mean,
                "tracking_variance": col_imu_var,
                "sensor_config": sensor_config,
                "high_freq_weight": 0.00,
            }
        },
        {
            "name": "effort_term",
            "weight": args.weight_effort,
            "args": {
                "exponent": 3,
                "speedweighting": True
            }
        }
    ]
    
            

    # Initialize optimal control problem
    print("Setting up Collocation OCP...")
    ocp = collocation.Collocation(model, config)
    ocp.setup()
    
    # Define dashboard persistence helper
    def keep_dashboard_alive():
        print("\n[INFO] Interactive OCP Dashboard is running at http://localhost:8050")
        print("Press Ctrl+C in this terminal to stop the server and exit.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down dashboard...")

    if args.no_solve:
        print("\n[INFO] --no-solve specified. Skipping OCP solver execution.")
        print("[INFO] Model parameters scaled successfully:")
        print(f"  Subject: {subject_id} (Mass: {mass} kg, Height: {height} m)")
        print(f"  Collocation successfully set up for {nnodes} nodes (full-cycle duration {dur_full:.4f} s).")
        print("[INFO] Setup and JAX compilation complete. Ready to track experimental trials!")
        return
        
    # Solve OCP
    print("Solving optimal control tracking problem with CyIPOPT...")
    solution = ocp.solve(visualize=True)
    tracked_states = solution.states
    tracked_globals = solution.globals
    info = solution.info
    
    print("\n=== Tracking OCP Solution Completed ===")
    print(f"IPOPT Status: {info.get('status_msg')}")
    print(f"Number of iterations: {info.get('iter')}")
    print(f"Final objective value: {info.get('obj_val'):.4f}")
    
    # 4. Extract results and verify tracking
    q_names = model.coordinates.names
    tracked_q = tracked_states.q
    tracked_qd = tracked_states.qd
    
    # Compute simulated acceleration qdd via finite difference
    N = tracked_qd.shape[0]
    if tracked_globals is not None and hasattr(tracked_globals, "h") and tracked_globals.h is not None and tracked_globals.h.size > 0:
        h_vals = tracked_globals.h.flatten()
    else:
        dt = dur_full / (N - 1)
        h_vals = np.ones(N - 1) * dt
    
    qdd_diff = (tracked_qd[1:] - tracked_qd[:-1]) / h_vals[:, None]
    tracked_qdd = np.concatenate([qdd_diff, qdd_diff[-1:]], axis=0)
    
    # Pad tau, ext_forces, ext_torques with zeros to form full model vector
    tau_zeros = np.zeros((N, model.tau.n))
    ef_zeros = np.zeros((N, model.ext_forces.n))
    et_zeros = np.zeros((N, model.ext_torques.n))
    
    states_model_batch = np.concatenate([tracked_q, tracked_qd, tracked_qdd, tau_zeros, ef_zeros, et_zeros], axis=1)

    # Instantiate sensor model with subject offsets to compute simulated proper IMU signals
    sensor_model = IMUSensorModel(model, config=sensor_config)
    sim_imu = np.array(sensor_model.compute_batch(
        states_model_batch,
        model.default_constants.filter('model').to_array()
    ))
    
    # Re-run names mapping
    sensor_names = sensor_model.get_sensor_names()
    
    # Plotting comparisons
    print("Generating validation and verification plots...")
    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    
    # Subplot 1: Hip Flexion Angle Reconstruction (Sim vs Exp)
    axs[0, 0].plot(t_col, np.rad2deg(q_hip_r_exp_col), 'r-', linewidth=2, label="Experimental (from angles)")
    axs[0, 0].plot(t_col, np.rad2deg(tracked_q[:, 3]), 'b--', linewidth=2, label="Reconstructed (IMU-only tracking)")
    axs[0, 0].set_title("Right Hip Flexion Angle", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Angle (deg)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Subplot 2: Knee Angle Reconstruction (Sim vs Exp)
    axs[1, 0].plot(t_col, np.rad2deg(q_knee_r_exp_col), 'r-', linewidth=2, label="Experimental (from angles)")
    axs[1, 0].plot(t_col, np.rad2deg(tracked_q[:, 4]), 'b--', linewidth=2, label="Reconstructed (IMU-only tracking)")
    axs[1, 0].set_title("Right Knee Angle", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Angle (deg)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Subplot 3: Ankle Angle Reconstruction (Sim vs Exp)
    axs[2, 0].plot(t_col, np.rad2deg(q_ankle_r_exp_col), 'r-', linewidth=2, label="Experimental (from angles)")
    axs[2, 0].plot(t_col, np.rad2deg(tracked_q[:, 5]), 'b--', linewidth=2, label="Reconstructed (IMU-only tracking)")
    axs[2, 0].set_title("Right Ankle Angle", fontsize=12, fontweight='bold')
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Angle (deg)")
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # Subplot 4: Pelvis vertical proper acceleration (Acc Y) tracking
    pelvis_acc_y_idx = sensor_names.index("pelvis_acc_y")
    axs[3, 0].plot(t_col, col_imu_mean[:, pelvis_acc_y_idx], 'r-', linewidth=2, label="Experimental (IMU)")
    axs[3, 0].plot(t_col, sim_imu[:, pelvis_acc_y_idx], 'b--', linewidth=2, label="Simulated")
    axs[3, 0].set_title("Pelvis Vertical Proper Acceleration", fontsize=12, fontweight='bold')
    axs[3, 0].set_xlabel("Time (s)")
    axs[3, 0].set_ylabel("Acceleration (m/s^2)")
    axs[3, 0].legend()
    axs[3, 0].grid(True)
    
    # Subplot 5: Tibia longitudinal proper acceleration (Acc X) tracking
    tibia_r_acc_x_idx = sensor_names.index("tibia_r_acc_x")
    axs[0, 1].plot(t_col, col_imu_mean[:, tibia_r_acc_x_idx], 'r-', linewidth=2, label="Experimental (IMU)")
    axs[0, 1].plot(t_col, sim_imu[:, tibia_r_acc_x_idx], 'b--', linewidth=2, label="Simulated")
    axs[0, 1].set_title("Right Tibia Longitudinal Acceleration", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Acceleration (m/s^2)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Subplot 6: Tibia transverse proper acceleration (Acc Y) tracking
    tibia_r_acc_y_idx = sensor_names.index("tibia_r_acc_y")
    axs[1, 1].plot(t_col, col_imu_mean[:, tibia_r_acc_y_idx], 'r-', linewidth=2, label="Experimental (IMU)")
    axs[1, 1].plot(t_col, sim_imu[:, tibia_r_acc_y_idx], 'b--', linewidth=2, label="Simulated")
    axs[1, 1].set_title("Right Tibia Transverse Acceleration", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Acceleration (m/s^2)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Subplot 7: Tibia pitch angular velocity (Gyr Z) tracking
    tibia_r_gyr_z_idx = sensor_names.index("tibia_r_gyr_z")
    axs[2, 1].plot(t_col, col_imu_mean[:, tibia_r_gyr_z_idx], 'r-', linewidth=2, label="Experimental (IMU)")
    axs[2, 1].plot(t_col, sim_imu[:, tibia_r_gyr_z_idx], 'b--', linewidth=2, label="Simulated")
    axs[2, 1].set_title("Right Tibia Angular Velocity (Pitch)", fontsize=12, fontweight='bold')
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Angular Velocity (rad/s)")
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    # Subplot 8: Convergence and muscle effort info
    axs[3, 1].axis('off')
    info_text = (
        f"Participant ID: {subject_id} (Height: {height:.2f}m, Mass: {mass:.1f}kg)\n"
        f"Trial Condition: {trial_type}\n"
        f"IPOPT Status: {info.get('status_msg')}\n"
        f"Number of Iterations: {info.get('iter')}\n"
        f"Final Objective Value: {info.get('obj_val'):.4f}\n"
        f"Collocation Nodes: {nnodes}\n"
        f"Gait Speed: {speed:.4f} m/s\n"
    )
    axs[3, 1].text(0.05, 0.5, info_text, fontsize=12, va='center', ha='left',
                  bbox=dict(boxstyle='round,pad=1', facecolor='aliceblue', edgecolor='lightsteelblue'))
    
    plt.tight_layout()
    if args.plot_file is None:
        plot_path = os.path.join(demo_dir, "result", f"{subject_id}_{trial_type}_results.png")
    else:
        plot_path = args.plot_file
        
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Validation and tracking comparison plot saved to {plot_path}")
    
    # Copy plot to app data artifacts folder if running inside agent workspace
    artifacts_dir = os.getenv("GEMINI_ARTIFACTS_DIR", "/Users/markusgambietz/.gemini/antigravity/brain/7dae6d19-5c37-44a7-9a83-afd1d0d4773d")
    if os.path.exists(artifacts_dir):
        dest_plot = os.path.join(artifacts_dir, f"{subject_id}_{trial_type}_results.png")
        import shutil
        try:
            shutil.copy2(plot_path, dest_plot)
            print(f"Plot copied to artifacts directory: {dest_plot}")
        except Exception as e:
            print(f"Warning: Could not copy plot to artifacts folder: {e}")
        
    print("\n=== Gait IMU-Only Optimal Control Tracking Complete ===")


if __name__ == "__main__":
    main()
