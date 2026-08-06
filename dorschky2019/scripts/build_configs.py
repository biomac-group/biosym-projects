"""
Generate one fully self-contained YAML config per participant per experiment
stage -- no placeholders, no runtime patching. Paths inside the project are
relative to the project root (PROJECT_DIR); scripts/run_experiment.py chdirs
there before loading a config, so these resolve regardless of the caller's
own working directory. Each file can then be run directly:

    cd dorschky2019  # or use run_experiment.py, which does this for you
    from biosym.ocp import collocation
    collocation.Collocation("configs/P02/ig_standing.yaml").solve()

Layout (mirrors result/{subject}/{stage}.pkl):
    configs/{subject}/ig_standing.yaml   -- standing equilibrium warm start
    configs/{subject}/ig_walking.yaml    -- population-average walking warm start
    configs/{subject}/ig_running.yaml    -- population-average running warm start
                                             (Fukuchi et al. 2017, 3.5 m/s), used as
                                             the initial guess for *running* trials
    configs/{subject}/{trial}.yaml       -- IMU tracking, one per trial type
                                             (slowwalking, normwalking, fastwalking use
                                              ig_walking.pkl; slowrunning, normrunning,
                                              fastrunning use ig_running.pkl)

Run scripts/build_participant_models.py first (generates models/participants/).

Objective weights below are placeholders pending confirmation against
Dorschky et al. 2019's published values -- see TODO markers.
"""
import os

import pandas as pd
import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "participants")
CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")
RESULT_DIR = os.path.join(PROJECT_DIR, "result")
DATA_DIR = os.path.join(PROJECT_DIR, "data")


def rel(path: str) -> str:
    """Path relative to PROJECT_DIR, for anything that lives inside this project."""
    return os.path.relpath(path, start=PROJECT_DIR)


WALKING_TRIAL_TYPES = ["slowwalking", "normwalking", "fastwalking"]
RUNNING_TRIAL_TYPES = ["slowrunning", "normrunning", "fastrunning"]
TRIAL_TYPES = WALKING_TRIAL_TYPES + RUNNING_TRIAL_TYPES

# Weights for the population-average warm-start walking/running stages (not
# part of the published method -- these stages only bootstrap a periodic gait
# for the IMU tracking stage below, so they remain pragmatic/tuned rather than
# paper values).
WEIGHT_EFFORT_IG = 300
WEIGHT_TRACK_ANGLES_IG = 1
WEIGHT_TRACK_GRF_IG = 1

# ig_running tracks a population-average running pattern derived from Fukuchi
# et al. (PeerJ, 2017) 3.5 m/s treadmill running data -- see
# build_ig_running_data.py. That source only yields hip/knee/GRF-x/y; pelvis
# translation/tilt has no data and is excluded from tracking below. Ankle
# angle also has no data there (tracked to a zero-mean placeholder) but is
# intentionally not excluded.
IG_RUNNING_ANGLE_EXCLUDE = ["pelvis_tx", "pelvis_ty", "pelvis_tilt"]

# Dorschky et al. 2019 objective (IMU tracking + effort): weight_tracking=1,
# weight_effort=300, effort exponent=2.
WEIGHT_IMU_TRACKING = 1
WEIGHT_EFFORT_TRACKING = 300
EFFORT_EXPONENT_TRACKING = 2


def dump(cfg: dict) -> str:
    return yaml.dump(cfg, sort_keys=False, default_flow_style=None)


def build_ig_standing(subject: str) -> dict:
    return {
        "collocation": {
            "name": f"{subject}_ig_standing",
            "description": "Single-node standing equilibrium warm start.",
            "settings": {
                "model": rel(os.path.join(MODELS_DIR, f"{subject}.yaml")),
                "nnodes": 1,
                "output": {"file": rel(os.path.join(RESULT_DIR, subject, "ig_standing.pkl"))},
                "tol": 5e-4,
                "max_iter": 3000,
            },
            "objectives": [
                {"name": "effort_term", "weight": 1.0, "args": {"exponent": 2}},
            ],
            "constraints": [
                {"name": "dynamics"},
            ],
            "bounds": {"from_model": True, "start_at_origin": True},
            "initial_guess": {"type": "random"},
        }
    }


def build_ig_walking(subject: str) -> dict:
    return {
        "collocation": {
            "name": f"{subject}_ig_walking",
            "description": (
                "100-node half-cycle walking warm start, tracking biosym's "
                "population-average joint angles/GRF (not subject/trial specific)."
            ),
            "settings": {
                "model": rel(os.path.join(MODELS_DIR, f"{subject}.yaml")),
                "nnodes": 100,
                "enable_logging": True,
                "iteration_log_interval": 20,
                "discretization": {
                    "type": "euler",
                    "args": {"mode": "backward", "vars": "q", "weight": 1, "adaptive_h": False},
                },
                "output": {"file": rel(os.path.join(RESULT_DIR, subject, "ig_walking.pkl"))},
                "tol": 5e-4,
                "acceptable_tol": 1e-3,
                "acceptable_dual_inf_tol": 1e-3,
                "constr_viol_tol": 1e-4,
                "max_iter": 4000,
            },
            "objectives": [
                {
                    "name": "track_angles",
                    "weight": WEIGHT_TRACK_ANGLES_IG,
                    "args": {
                        "presegmented": True,
                        "presegmented_file": rel(os.path.join(DATA_DIR, "gait_avg_joint_angles.mot")),
                        "exclude": ["pelvis_tx", "pelvis_ty"],
                    },
                },
                {"name": "effort_term", "weight": WEIGHT_EFFORT_IG, "args": {"exponent": 3, "speedweighting": True}},
                {
                    "name": "track_grf",
                    "weight": WEIGHT_TRACK_GRF_IG,
                    "args": {"presegmented": True, "presegmented_file": rel(os.path.join(DATA_DIR, "gait_avg_grf.mot"))},
                },
            ],
            "constraints": [
                {"name": "dynamics"},
                {"name": "periodicity", "args": {"symmetry": False, "exclude": [0]}},
            ],
            "initial_guess": {"type": "from_file", "file": rel(os.path.join(RESULT_DIR, subject, "ig_standing.pkl"))},
            "bounds": {"from_model": True, "start_at_origin": True, "dur": [0.5, 1.3], "speed": 1.3},
        }
    }


def build_ig_running(subject: str) -> dict:
    return {
        "collocation": {
            "name": f"{subject}_ig_running",
            "description": (
                "100-node half-cycle running warm start, tracking a population-average "
                "running pattern derived from Fukuchi et al. (2017) 3.5 m/s treadmill "
                "running data (not subject/trial specific)."
            ),
            "settings": {
                "model": rel(os.path.join(MODELS_DIR, f"{subject}.yaml")),
                "nnodes": 100,
                "enable_logging": True,
                "iteration_log_interval": 20,
                "discretization": {
                    "type": "euler",
                    "args": {"mode": "backward", "vars": "q", "weight": 1, "adaptive_h": False},
                },
                "output": {"file": rel(os.path.join(RESULT_DIR, subject, "ig_running.pkl"))},
                "tol": 5e-4,
                "acceptable_tol": 1e-3,
                "acceptable_dual_inf_tol": 1e-3,
                "constr_viol_tol": 1e-4,
                "max_iter": 4000,
            },
            "objectives": [
                {
                    "name": "track_angles",
                    "weight": WEIGHT_TRACK_ANGLES_IG,
                    "args": {
                        "presegmented": True,
                        "presegmented_file": rel(os.path.join(DATA_DIR, "fukuchi_running_joint_angles.mot")),
                        "exclude": IG_RUNNING_ANGLE_EXCLUDE,
                    },
                },
                {"name": "effort_term", "weight": WEIGHT_EFFORT_IG, "args": {"exponent": 3, "speedweighting": True}},
                {
                    "name": "track_grf",
                    "weight": WEIGHT_TRACK_GRF_IG,
                    "args": {"presegmented": True, "presegmented_file": rel(os.path.join(DATA_DIR, "fukuchi_running_grf.mot"))},
                },
            ],
            "constraints": [
                {"name": "dynamics"},
                {"name": "periodicity", "args": {"symmetry": False, "exclude": [0]}},
            ],
            "initial_guess": {"type": "from_file", "file": rel(os.path.join(RESULT_DIR, subject, "ig_walking.pkl"))},
            "bounds": {"from_model": True, "start_at_origin": True, "dur": [0.4, 1.0], "speed": 3.5},
        }
    }


def build_trial(subject: str, trial: str, dur_full: float) -> dict:
    return {
        "collocation": {
            "name": f"{subject}_{trial}",
            "description": f"Muscle-driven IMU-only gait tracking for {subject}/{trial} (Dorschky et al. 2019).",
            "settings": {
                "model": rel(os.path.join(MODELS_DIR, f"{subject}.yaml")),
                "nnodes": 100,
                "enable_logging": True,
                "log_interval": 10,
                "discretization": {
                    "type": "euler",
                    "args": {"mode": "backward", "vars": "q", "weight": 1, "adaptive_h": False},
                },
                "output": {"file": rel(os.path.join(RESULT_DIR, subject, f"{trial}.pkl"))},
                "tol": 5e-4,
                "acceptable_dual_inf_tol": 1e-3,
                "acceptable_tol": 1e-3,
                "constr_viol_tol": 1e-4,
                "max_iter": 8000,
            },
            "objectives": [
                {
                    "name": "IMUTrackingObjective",
                    "weight": WEIGHT_IMU_TRACKING,
                    "args": {"subject": subject, "trial": trial, "data_dir": rel(DATA_DIR)},
                },
                {
                    "name": "effort_term",
                    "weight": WEIGHT_EFFORT_TRACKING,
                    "args": {"exponent": EFFORT_EXPONENT_TRACKING, "speedweighting": True},
                },
            ],
            "constraints": [
                {"name": "dynamics"},
                {"name": "periodicity", "args": {"symmetry": False, "exclude": [0]}},
            ],
            "initial_guess": {
                "type": "from_file",
                "file": rel(os.path.join(RESULT_DIR, subject, "ig_running.pkl" if trial in RUNNING_TRIAL_TYPES else "ig_walking.pkl")),
            },
            "bounds": {"from_model": True, "start_at_origin": True, "dur": [dur_full, dur_full], "speed": [0, 10]},
        }
    }


def main():
    df_info = pd.read_csv(os.path.join(DATA_DIR, "ParticipantInfo.csv"))
    for subject in df_info["ID"]:
        subject_dir = os.path.join(CONFIGS_DIR, subject)
        os.makedirs(subject_dir, exist_ok=True)
        os.makedirs(os.path.join(RESULT_DIR, subject), exist_ok=True)

        with open(os.path.join(subject_dir, "ig_standing.yaml"), "w") as f:
            f.write(dump(build_ig_standing(subject)))
        with open(os.path.join(subject_dir, "ig_walking.yaml"), "w") as f:
            f.write(dump(build_ig_walking(subject)))
        with open(os.path.join(subject_dir, "ig_running.yaml"), "w") as f:
            f.write(dump(build_ig_running(subject)))

        for trial in TRIAL_TYPES:
            parquet_path = os.path.join(DATA_DIR, f"{subject}_{trial}_mean_FP.mat.parquet")
            if not os.path.exists(parquet_path):
                print(f"  Skipping {subject}/{trial}: no data file at {parquet_path}")
                continue
            df_pq = pd.read_parquet(parquet_path)
            dur_full = float(df_pq[df_pq["type"] == "dur"].iloc[0]["mean"][0])
            with open(os.path.join(subject_dir, f"{trial}.yaml"), "w") as f:
                f.write(dump(build_trial(subject, trial, dur_full)))

        print(f"Wrote configs for {subject}")


if __name__ == "__main__":
    main()
