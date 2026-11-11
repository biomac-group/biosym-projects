# Dorschky et al. (2019) reproduction -- validation report

Accuracy and runtime of the completed muscle-driven IMU-tracking trials, against each subject's own optical-motion-capture (OMC) + force-plate reference data, and against the Dorschky et al. (2019) OCP baseline as compiled by SSPINNpose (Gambietz et al., *SSPINNpose: A Self-Supervised PINN for Inertial Pose and Dynamics Estimation*, ACM TIST, 2026).

## Comparison to Dorschky et al. (2019) / SSPINNpose baseline

| Metric | This reproduction | Dorschky (2019) |
|---|---|---|
| Joint angle RMSD (deg) | 7.93 | 6.3 |
| Speed RMSD (m/s) | 0.29 | 0.25 |
| GRF RMSD (%BW) | 14.8 | 17.9 |
| Joint torque RMSD (%BWBH) | 2.1 | 2.6 |
| Runtime, Intel Xeon E3-1240 (2017) [min] | 2.5 +/- 0.9* | 50 +/- 26 |
| Runtime, AMD Ryzen 9 9950X (2024) [min] | 0.6 +/- 0.2* | n/a |
*multi-threaded biosym, single-threaded IPOPT (via `OMP_NUM_THREADS=1`, which substantially reduces IPOPT runtime for this problem)

Minor differences are expected due to some differences in the implementation:
* Different initial guess data for walking / running trials; Dorschky et al. did not specify their data source, so we used self-recorded data at 1.3 m/s for walking and data from Fukuchi et al. (2018) for running at 3.5 m/s.
* Ground contact model: Dorschky et al. used a linear spring-damper contact model, while we used a nonlinear Hunt-Crossley contact model. (We use Hunt-Crossley with contact points instead of spheres, which will be deprecated in future releases of biosym, because the forces should technically be zero when a point with r=0 is used.)
* Additionally, minor mathematical differences in the implementation of the constraints can also have an effect on the solution, however, we expect that when re-running these trials, the results may differ slightly when using a different initial guess, and in general, the results are expected to be similar to those reported in Dorschky et al. (2019). The differences also depend on hardware & IPOPT version: our Intel Xeon E3-1240 run reports: JAE: 7.47, Speed Error: 0.23, GRF Error: 9.7, JTE: 1.7.
* We don't use Dorschky et al.'s "Regularization" term.

## Validation plots

Plots compare our results (blue) vs. reference OMC + Force Plate data (grey)

### Slow Walking

n=10 for joint angle/speed, n=10 for moment/GRF.

![slowwalking validation grid](result/report_figures/validation_slowwalking.png)

### Normal Walking

n=10 for joint angle/speed, n=10 for moment/GRF.

![normwalking validation grid](result/report_figures/validation_normwalking.png)

### Fast Walking

n=10 for joint angle/speed, n=10 for moment/GRF.

![fastwalking validation grid](result/report_figures/validation_fastwalking.png)

### Slow Running

n=10 for joint angle/speed, n=10 for moment/GRF.

![slowrunning validation grid](result/report_figures/validation_slowrunning.png)

### Normal Running

n=10 for joint angle/speed, n=10 for moment/GRF.

![normrunning validation grid](result/report_figures/validation_normrunning.png)

### Fast Running

n=10 for joint angle/speed, n=10 for moment/GRF.

![fastrunning validation grid](result/report_figures/validation_fastrunning.png)
