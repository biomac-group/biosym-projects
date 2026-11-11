# biosym_projects

Reproduction/reimplementation projects built on top of the `biosym` package
(a JAX-based direct-collocation optimal control framework for musculoskeletal
simulation). Each subdirectory is an independent project with its own data,
models, and pipeline; `biosym` itself is installed as a dependency, not
vendored into any of them.

## Reimplementations

- **[`dorschky2019/`](dorschky2019/)** -- reproduction of Dorschky, E.,
  Nitschke, M., Seifer, A. K., van den Bogert, A. J., & Eskofier, B. M.
  (2019), *Estimation of gait kinematics and kinetics from inertial sensor
  data using optimal control of musculoskeletal models*, Journal of
  Biomechanics, 95, 109278. A planar 7-segment, 16-muscle musculoskeletal
  model that tracks per-participant IMU (accelerometer/gyroscope) data
  across walking and running speeds. See `dorschky2019/README.md` for the
  full pipeline, layout, and usage, and `dorschky2019/REPORT.md` for current
  validation results against the original paper's numbers.

- **[`3d_simulations/`](3d_simulations/)** -- 3D gait prediction without tracking data, the biosym version of the
  BiomechPriorVAE paper pipeline ([paper](https://doi.org/10.48550/arXiv.2610.08506),
  [simulation repository](https://github.com/gambimar/BiomechPriorVAE),
  [figures and website](https://github.com/gambimar/BiomechPriorVAE-visuals)). BioMAC's SIPP runmad model (33 DOF,
  92 Gait3d muscles, 10 arm torque actuators, 12 implicit Hunt-Crossley contact spheres) walks or runs a half stride at a
  set or self-chosen speed from a random standing pose; the objectives are the Bhargava cost of transport (or muscle
  effort), a learned state prior, arm effort and a heel-strike variance penalty. Results are compared with the paper's
  reference data and baselines in the biosym versions of its Figures 1, 3 and 4. `python 3d_simulations/run.py 1.2`
  runs one simulation, see `3d_simulations/README.md`.

- **[`imu_uncertainty/`](imu_uncertainty/)** -- learned, state-dependent IMU measurement uncertainty and a learned
  pose prior for IMU-driven optimal-control gait estimation, built on `dorschky2019`. See `imu_uncertainty/README.md`.
 
