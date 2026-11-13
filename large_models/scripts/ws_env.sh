# Environment of the sweep on the workstation (sourced by ws_run_sweep.sh): the new biosym tree, one thread per solve.
export PYTHONPATH=$HOME/phd/biosym_myofull/biosym:$HOME/phd/biosym_myofull/large_models
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BIOSYM_DISABLE_CACHE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PY=$HOME/phd/biosym-projects/dorschky2019/.micromamba/root/envs/dorschky-cyipopt/bin/python
