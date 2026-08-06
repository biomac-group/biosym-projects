from scripts import build_configs, build_participant_models
import subprocess

if __name__ == "__main__":
    # Build participant models (if not already present)
    build_participant_models.main()
    build_configs.main()    

    for participants in ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]:
        subprocess.run(["python", "scripts/run_experiment.py", participants, "ig_standing"])
        subprocess.run(["python", "scripts/run_experiment.py", participants, "ig_walking"])
        subprocess.run(["python", "scripts/run_experiment.py", participants, "ig_running"])

        for trial in build_configs.TRIAL_TYPES:
            print(f"Running {participants}/{trial}...")
            subprocess.run(["python", "scripts/run_experiment.py", participants, trial])