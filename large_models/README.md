# large_models: MS-Human-700 and MyoFullBody in biosym

Scripts based on `biosym==0.2.0`.

Predictive 3D walking at 1.3 m/s with two large MuJoCo musculoskeletal models in `biosym`:

| Model | Source | biosym model | Muscles |
|---|---|---|---|
| **MS-Human-700** (`ms_human_700`) | MuJoCo Menagerie port of LNSGroup's model | 43 coordinates, 38 bodies, 60.2 kg | 700 |
| **MyoFullBody** (`myofullbody`) | MyoHub `myo_sim` (`compose --generate`) | 77 coordinates (incl. 40 finger joints), 69 bodies, 84.3 kg | 416 |

The problem is the one from [`biosym_projects/3d_simulations`](https://github.com/biomac-group/biosym-projects/tree/main/3d_simulations). Both models use MuJoCo's own muscle (rigid tendon) with MuJoCo's activation dynamics
(an excitation state per muscle), as published.

Neither model has an OpenSim version. The scripts compile the MJCF with the `mujoco` package and write biosym-format
files, fit the muscle paths to MuJoCo's tendon geometry, and check the result against MuJoCo itself. 
`mujoco` is only needed to (re)make the model files.

```bash
python run.py ms_human_700        # standing (once, seed 1) -> walking at 1.3 m/s -> viewer (port 8080)
python run.py myofullbody         # the same with MyoFullBody (viewer port 8082, dashboard 8051)
python run.py myofullbody metabolic   # metabolic cost of transport instead of muscle effort (from the effort solution)
```

On a MacBook Pro M1, the walking optimization takes less than 1 hour for most seeds. 

Make a model (MyoFullBody first needs `uv run python -m myo_sim.build.compose --generate` in
`models/original/myo_sim`), about 6 to 10 minutes, most of it the path fits:

```bash
uv run --with "mujoco>=3.4" python scripts/pair_muscles.py MODEL    # muscle names with uniform _r/_l (muscle_names.csv)
uv run --with "mujoco>=3.4" python scripts/make_model.py MODEL      # models/MODEL/*: bodies, muscles, yaml, fit report
uv run --with "mujoco>=3.4" python scripts/make_contact.py MODEL    # levelled foot spheres (MyoFullBody; MS-Human's are hand-made)
uv run --with "mujoco>=3.4" python scripts/make_visuals.py MODEL    # meshes + muscle lines for the viewer (visuals.npz)
uv run --with "mujoco>=3.4" python scripts/check_model.py MODEL     # kinematics, dynamics, muscles, mirror, visuals vs MuJoCo
```

## Layout

```
run.py                   standing from make_prior_guess, then walking, then view();  python run.py MODEL
configs/                 MODEL_standing.yaml, MODEL_walking.yaml
objectives/              prior_map.py: the state prior with the model -> gait3d coordinate map (thorax body as argument)
models/original/         ms_human_700/ (MuJoCo Menagerie, Apache-2.0), myo_sim/ (MyoHub, Apache-2.0); SOURCE.txt each
models/MODEL/            the biosym model: MODEL.yaml/.xml, MODEL_muscles.xml, contact_implicit.xml, muscle_names.csv,
                         fit_report.csv, visuals.npz
scripts/                 model_config.py (the models), pair_muscles.py, make_model.py, make_contact.py, make_visuals.py,
                         check_model.py
results/MODEL/           standing_seed1.pkl, walking_v1.30_seed1.pkl; MS-Human also the rendered video (.mp4, .png)
logs/                    IPOPT logs and the logs of the scripts
```

## Model conventions

y up, x forward, z right; the floor is y = 0. The world is the pelvis frame of the MuJoCo default pose (MyoFullBody's,
which is z up in MuJoCo, re-expressed and levelled by 0.046 deg), so biosym's `pelvis_ty` is the MuJoCo pelvis height
(0.95 m MS-Human, 1.0 m MyoFullBody) in the default pose. A free root joint becomes `pelvis_tx/ty/tz`, `pelvis_tilt` (z),
`pelvis_list` (x), `pelvis_rotation` (y). The knee is positive in flexion. Coordinate names are the models' own; muscle
names and four MyoFullBody hand joints get a uniform `_r`/`_l` suffix (the mirror symmetry pairs by it).

## Models and citations

- Zuo, C., He, K., Shao, J., & Sui, Y. (2024). *Self model for embodied intelligence: Modeling full-body human
  musculoskeletal system and locomotion control with hierarchical low-dimensional representation.* ICRA 2024,
  13062-13069. Upstream: <https://github.com/LNSGroup/MS-Human-700>; MJCF from MuJoCo Menagerie.
- MyoSuite / MyoSim: Caggiano, V., Wang, H., Durandau, G., Sartori, M., & Kumar, V. (2022). *MyoSuite: A contact-rich
  simulation suite for musculoskeletal motor control.* L4DC 2022. Models: <https://github.com/MyoHub/myo_sim>.
