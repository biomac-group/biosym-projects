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

- **`3d_simulations/`** -- stay tuned :) 
