# dorschky2019

Reproduction of the muscle-driven IMU-tracking gait reconstruction from
Dorschky, E., Nitschke, M., Seifer, A. K., van den Bogert, A. J., &
Eskofier, B. M. (2019), *Estimation of gait kinematics and kinetics from
inertial sensor data using optimal control of musculoskeletal models*,
Journal of Biomechanics, 95, 109278, built on top of the `biosym` package
(installed as a dependency, not vendored into this project).

Per-participant walking/running trial data are from the same study
(DOI 10.5281/zenodo.11522049).

## Pipeline

For each participant, three warm-start stages bootstrap a periodic gait
before the actual IMU-tracking optimization:

1. **`ig_standing`** -- single-node standing equilibrium, minimal effort,
   random initial guess.
2. **`ig_walking`** -- 100-node walking warm start, tracking a
   population-average (not subject-specific) walking joint-angle/GRF
   pattern (`data/gait_avg_joint_angles.mot`, `data/gait_avg_grf.mot`),
   initialized from `ig_standing`.
3. **`ig_running`** -- 100-node running warm start, tracking a
   population-average running pattern derived from Fukuchi et al. (PeerJ,
   2017) 3.5 m/s treadmill running data (`data/fukuchi_running_joint_angles.mot`,
   `data/fukuchi_running_grf.mot`), initialized from `ig_walking`.
   Pelvis translation/tilt has no data in that source and is excluded from
   tracking there. Both stages
   track cross-subject/cross-cycle population variance; GRF variance is floored at 1 N^2 (the raw Fukuchi cross-subject GRF variance was too low, overweighting the GRF term).

The actual IMU-tracking trials (`slowwalking`, `normwalking`,
`fastwalking`, `slowrunning`, `normrunning`, `fastrunning`) then track each
participant's real IMU data, using `ig_walking` as the initial guess for
walking trials and `ig_running` for running trials.

All data referenced by generated configs can be found under `data/` inside this project.

## Layout

```
data/                      Per-participant trial data (IMU + force plate),
                            population-average walking/running reference
                            data, and participant metadata (ParticipantInfo.csv).
models/                    gait2d template model + per-participant scaled models
                            (models/participants/, generated).
configs/{subject}/         One self-contained YAML per stage/trial, generated
                            by scripts/build_configs.py (see below).
result/{subject}/          Solver outputs (.pkl), one per stage/trial.
scripts/
  build_participant_models.py  Scale the template model to each participant's
                                height/mass -> models/participants/.
  build_ig_running_data.py     Generate data/fukuchi_running_*.mot from the raw
                                Fukuchi subject files in data/fukuchi_running_raw/.
  build_configs.py             Generate configs/{subject}/*.yaml (run after the
                                two scripts above).
  run_experiment.py            Run a single config: `python scripts/run_experiment.py P02 ig_running`.
src/                       Project-specific objective (IMUTrackingObjective) and
                            IMU data loading, registered into biosym.ocp.objfun
                            by run_experiment.py.
run_all.py                 Build everything and run all stages/trials for all
                            10 participants, end to end.
```

## Usage

To run a single participant's stages/trials, e.g. P02's normal walking and running trials, and showing the validation video, run:
```bash
cd dorschky2019
python scripts/build_participant_models.py
python scripts/build_configs.py
python scripts/run_experiment.py P02 ig_standing
python scripts/run_experiment.py P02 ig_walking
python scripts/run_experiment.py P02 ig_running
python scripts/run_experiment.py P02 normwalking --visualize
python scripts/run_experiment.py P02 normrunning --visualize
```

or, to run everything for all participants and re-generate the validation report, run:

```bash
python run_all.py
```
