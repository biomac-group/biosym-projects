"""
Shared pieces of the BiomechPriorVAE paper pipeline: ``paper_standing.py`` and ``paper_walking.py``.

Both scripts use the standard 3D model (``models/sipp_runmad/sipp_runmad.yaml``). The optimisation problems are built
once per process (the model build, the compiled functions and the Jacobian structure are the expensive part) and
solved from many initial guesses with ``ocp.nlp.solve(x0)``.
"""
import os
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import cloudpickle  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)      # the configs refer to models and results relative to the project folder

from biosym.ocp import collocation  # noqa: E402
from biosym.ocp.utils import initial_guess as initial_guess_utils  # noqa: E402
from biosym.ocp.utils import vectorize  # noqa: E402
from biosym.utils import state_prior  # noqa: E402

RESULT_DIR = Path("results")
STANDING_TEMPLATE = Path("configs/standing_paper.yaml")
WALKING_TEMPLATE = Path("configs/walking.yaml")

# Activation every muscle is seeded at before the equilibrium solve: quiet standing needs a few percent, and a = 0
# makes the force-length term vanish.
SEED_ACTIVATION = 0.02
# Bracket of the normalised fibre length searched by the equilibrium bisection.
LCE_MIN, LCE_MAX = 0.3, 2.5
# Pelvis heights searched for the standing equilibrium [m].
HEIGHT_MIN, HEIGHT_MAX, HEIGHT_STEP = 0.3, 1.5, 0.01
# Bounds of the foot-force features [Fy_r, Flat_r, Fy_l, Flat_l] in body weights during the pose optimisation.
FORCE_BOUNDS = ((0.0, 1.5), (0.0, 0.5), (0.0, 1.5), (0.0, 0.5))


class Stopwatch:
    """Prints the time since its creation next to each message, so that a slow stage is visible."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.t_last = self.t0

    def __call__(self, message):
        now = time.perf_counter()
        print(f"[{now - self.t0:8.1f} s, +{now - self.t_last:6.1f}] {message}", flush=True)
        self.t_last = now


def write_config(template, out_path, edit):
    """
    Write a copy of a collocation config with ``edit`` applied to its ``collocation`` dictionary.

    Parameters
    ----------
    template : str or Path
        Config to copy.
    out_path : str or Path
        Where the copy is written (the folder is created).
    edit : callable
        Called with the ``collocation`` dictionary; changes it in place.

    Returns
    -------
    pathlib.Path
        ``out_path``.
    """
    cfg = yaml.safe_load(Path(template).read_text())
    edit(cfg["collocation"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


class ReusableProblem:
    """
    A collocation problem built once and solved from many initial guesses.

    Parameters
    ----------
    config : str or Path
        Collocation config. Its initial guess only fixes the layout; each solve passes its own ``x0``.
    """

    def __init__(self, config):
        self.ocp = collocation.Collocation(str(config))
        self.model = self.ocp.model
        self.settings = self.ocp.settings
        self.setup_seconds = self.ocp.wall_time_setup

    def x0(self, states, globals_=None):
        """Flat optimisation vector of structured states (and globals for problems with more than one node)."""
        return np.asarray(vectorize.states_dict_to_x(states, globals_ if self.settings["nnodes"] > 1 else None))

    def x0_from_file(self, path):
        """Initial guess of a result file, tiled over the nodes like ``initial_guess: from_file``."""
        states, globals_ = initial_guess_utils.make_initial_guess(
            self.model, self.settings, {"type": "from_file", "file": str(path)})
        return self.x0(states, globals_)

    def solve(self, x0, out=None):
        """
        Solve from ``x0`` with the existing NLP.

        Parameters
        ----------
        x0 : array-like
            Flat initial guess.
        out : str or Path, optional
            Result file, written in the format of ``Collocation.solve`` (``((states, globals), info, settings)``).

        Returns
        -------
        dict
            ``status``, ``converged`` (status 0), ``objective``, ``terms`` (objective terms), ``solve_seconds``,
            ``x`` (structured states and globals) and the IPOPT ``info``.
        """
        t0 = time.perf_counter()
        x, info = self.ocp.nlp.solve(np.asarray(x0))
        seconds = time.perf_counter() - t0
        info["wall_time_solve"] = seconds
        info["wall_time_setup"] = self.setup_seconds
        x_states = vectorize.x_to_states_dict(
            x, self.ocp.initial_guess_states,
            self.ocp.initial_guess_globals if self.settings["nnodes"] > 1 else None)
        terms = {k: float(v) for k, v in self.ocp._evaluate_objective_values(*x_states).items()}
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                cloudpickle.dump((x_states, info, self.settings), f)
        return {"status": int(info["status"]), "converged": int(info["status"]) == 0,
                "objective": float(info["obj_val"]), "terms": terms, "solve_seconds": seconds,
                "x": x_states, "info": info}


# ----------------------------------------------------------------------------------------------------------------
# standing: initial guess of a seed
# ----------------------------------------------------------------------------------------------------------------

def prior_columns(model, prior):
    """Positions of the 23 prior coordinates in the model's coordinate vector and of the knees among them."""
    names = [str(n).removeprefix("q_") for n in model.coordinates.names]
    cols = np.array([names.index(n) for n in prior["coord_names"]], dtype=int)
    knees = [prior["coord_names"].index(n) for n in prior["knee_names"]]
    return cols, knees


def optimise_pose_on_prior(model, q_lo, q_hi, rng, maxiter=500):
    """
    Optimise joint angles and foot forces on the learned state prior, with zero joint velocities.

    A random pose and random foot forces are drawn and then minimise the prior's reconstruction NLL over the 23 prior
    coordinates (within their bounds) and the four foot-force features ``[Fy_r, Flat_r, Fy_l, Flat_l]`` (body weights).
    Velocities are fixed at zero. The pelvis coordinates are not part of the prior and are not touched.

    Parameters
    ----------
    model : BiosymModel
        Model of the standing problem.
    q_lo, q_hi : array-like of float
        Bounds of the model's coordinates (all of them), in the model's coordinate order.
    rng : numpy.random.Generator
        Source of the random start.
    maxiter : int, optional
        L-BFGS-B iteration limit.

    Returns
    -------
    dict
        ``q`` (full coordinate vector; the coordinates that are not in the prior, such as the pelvis, are zero), ``force``
        (the four force features), ``nll_start`` and ``nll_end``.
    """
    prior = state_prior.load_state_prior()
    cols, knees = prior_columns(model, prior)
    params, meta = prior["params"], prior["meta"]
    sign = np.ones(len(cols))
    sign[knees] = -1.0
    lo = np.concatenate([np.asarray(q_lo)[cols], [b[0] for b in FORCE_BOUNDS]])
    hi = np.concatenate([np.asarray(q_hi)[cols], [b[1] for b in FORCE_BOUNDS]])

    def nll(v):
        features = jnp.concatenate([v[:len(cols)] * sign, jnp.zeros(len(cols)), v[len(cols):]])
        return state_prior.nll_per_node(features[None, :], params, meta)[0]

    value_and_grad = jax.jit(jax.value_and_grad(nll))

    def fun(v):
        f, g = value_and_grad(jnp.asarray(v))
        return float(f), np.asarray(g, dtype=float)

    x_start = rng.uniform(lo, hi)
    x_start[len(cols):] = rng.uniform([0.2, 0.0, 0.2, 0.0], [0.8, 0.1, 0.8, 0.1])
    res = minimize(fun, x_start, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": maxiter})
    q = np.zeros(model.coordinates.n)
    q[cols] = res.x[:len(cols)]
    return {"q": q, "force": res.x[len(cols):], "nll_start": float(fun(x_start)[0]), "nll_end": float(res.fun)}


def vertical_force_function(model):
    """
    Function of the coordinate vector giving the total vertical contact force of the contact law at rest [body weights].

    The implicit contact model exposes the law as ``model.gc_model.law``; the force on every contact body is summed.
    """
    node = model.default_states
    constants = model.default_constants

    @jax.jit
    def vertical(q):
        state = node.replace(q=q, qd=jnp.zeros_like(node.qd))
        return jnp.sum(model.gc_model.law(state, constants, model)[:, 1])

    return vertical


def solve_pelvis_height(vertical, q, height_index, target=1.0, tol=1e-6):
    """
    Pelvis height at which the contact law carries ``target`` body weights.

    The vertical force falls monotonically with the pelvis height, so an upward scan brackets the crossing and
    bisection refines it.

    Parameters
    ----------
    vertical : callable
        From :func:`vertical_force_function`.
    q : numpy.ndarray
        Coordinate vector; its pelvis height entry is replaced.
    height_index : int
        Position of ``pelvis_ty`` in ``q``.
    target : float, optional
        Total vertical force in body weights.
    tol : float, optional
        Relative tolerance of the force.

    Returns
    -------
    tuple of float
        Height [m] and the force reached [body weights].
    """
    def force_at(height):
        qq = np.array(q, dtype=float)
        qq[height_index] = height
        return float(vertical(jnp.asarray(qq)))

    low = high = None
    for height in np.arange(HEIGHT_MIN, HEIGHT_MAX, HEIGHT_STEP):
        if force_at(float(height)) >= target:
            low = float(height)
        elif low is not None:
            high = float(height)
            break
    if low is None or high is None:
        raise RuntimeError(f"the contact law does not cross {target} body weights for pelvis_ty in "
                           f"[{HEIGHT_MIN}, {HEIGHT_MAX}] m")
    for _ in range(100):
        mid = 0.5 * (low + high)
        force = force_at(mid)
        if abs(force - target) < tol * target:
            return mid, force
        low, high = (mid, high) if force > target else (low, mid)
    mid = 0.5 * (low + high)
    return mid, force_at(mid)


def muscle_model(model):
    """The Hill-type member of the model's actuator model (``None`` without muscles)."""
    actuator = getattr(model, "actuator_model", None)
    for member in getattr(actuator, "models", None) or ([actuator] if actuator is not None else []):
        if hasattr(member, "muscle_equations"):
            return member
    return None


def equilibrate_muscles(model, q, activation=SEED_ACTIVATION, n_iter=80):
    """
    Fibre lengths at which every muscle is in force equilibrium at pose ``q`` and the given activation.

    The residual of the single-node constraint, ``(F_see - F_ce - F_pee) / F_max`` at zero fibre velocity, falls
    monotonically with the fibre length and depends only on the muscle's own length, so all muscles are bisected in
    lockstep.

    Parameters
    ----------
    model : BiosymModel
        Model with a Hill-type muscle member.
    q : numpy.ndarray
        Coordinate vector.
    activation : float, optional
        Activation and excitation every muscle is seeded at.
    n_iter : int, optional
        Bisection steps.

    Returns
    -------
    tuple
        The model's full ``actuator_model`` state vector (fibre lengths, activations, excitations; other members zero),
        the largest remaining residual in units of ``F_max``, and the fibre lengths.
    """
    muscles = muscle_model(model)
    n = muscles.n_actuators
    idx_lce, idx_a, idx_e = (np.asarray(muscles.idx[k]) for k in ("Lce", "a", "e"))
    f_max = np.asarray(muscles.muscle_constants["fmax"]).reshape(-1)
    template = model.default_states
    size = template.actuator_model.shape[0]

    def block(lce):
        b = np.zeros(size)
        b[idx_lce], b[idx_a], b[idx_e] = lce, activation, activation
        return b

    def residual(lce):
        state = template.replace(q=jnp.asarray(q), qd=jnp.zeros_like(template.qd), actuator_model=jnp.asarray(block(lce)))
        f_ce, f_see, f_pee = muscles.muscle_equations(state, jnp.zeros(n), model.default_constants, model)[:3]
        return np.asarray(f_see - f_ce - f_pee).reshape(-1) / f_max

    low, high = np.full(n, LCE_MIN), np.full(n, LCE_MAX)
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        positive = residual(mid) > 0             # the root lies at a longer fibre
        low, high = np.where(positive, mid, low), np.where(positive, high, mid)
    lce = 0.5 * (low + high)
    return block(lce), float(np.abs(residual(lce)).max()), lce
