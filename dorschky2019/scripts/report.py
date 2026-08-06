"""
Generate REPORT.md: joint-angle, joint-moment, GRF, and speed accuracy of the
completed IMU-tracking trials against each subject's own OMC-based reference
data, compared to the Dorschky et al. (2019) baseline numbers as compiled by
SSPINNpose (Gambietz et al.), plus a runtime figure and one validation plot
per trial condition.

GRF/joint-moment reconstruction: these aren't part of the solved state (only
q, qd, and the actuator/contact model's own internal states are), so they're
recomputed by re-evaluating that participant's *current* contact and actuator
force laws on the solved q/qd/actuator-state trajectory. That only makes
sense for trials solved with the code currently checked out -- for an older
result solved with a since-changed force law, re-evaluating today's law on
its trajectory can produce physically implausible output (e.g. negative
vertical GRF, i.e. the ground pulling the foot down). Each trial's
reconstruction is sanity-checked (vertical GRF must stay non-negative within
a small tolerance) and dropped from the GRF/moment panels -- but not the
joint-angle ones, which don't depend on any force law -- if it fails.

Nothing about which trials exist, which q-column is which joint, or measured
runtimes is hardcoded here: trials are discovered from `result/`, joint
indices and forces come from each participant's own model, and runtimes are
read from `result/runtime_measurements.json` (written by
`scripts/measure_runtime.py`) if present.

Usage:
    python scripts/report.py
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)

# Joints compared against the (right-side-only) OMC reference data.
JOINTS = ("hip_r", "knee_r", "ankle_r")

# Minimum tolerated vertical GRF (%BW) before a trial's reconstructed
# GRF/moments are considered physically implausible and dropped from those
# panels (see module docstring). Small negative numerical noise is fine;
# large sustained negative "the ground pulls the foot down" values are not.
MIN_PLAUSIBLE_VERTICAL_GRF_PCTBW = -10.0

# Dorschky et al. (2019) OCP baseline, as compiled by SSPINNpose (Gambietz et
# al., Table 3): joint angle error, joint torque error, GRF error, speed
# error, all ensemble-averaged-gait-cycle evaluation (matching the evaluation
# protocol used here).
REFERENCE = {
    "JAE_deg": 6.3,
    "JTE_pctBWBH": 2.6,
    "GRFE_pctBW": 17.9,
    "speed_error_ms": 0.25,
}
# Dorschky et al. (2019), Sec. 3 ("Results"): "60 optimizations (ten subjects
# at six speeds) were solved with a mean CPU time of 50 +/- 26 min on Intel
# Xeon processors E3-1240."
REFERENCE_RUNTIME_MEAN_MIN = 50
REFERENCE_RUNTIME_SD_MIN = 26
REFERENCE_RUNTIME_HARDWARE = "Intel Xeon E3-1240"

RUNTIME_FILE = "result/runtime_measurements.json"
MAX_OFFSET_NODES = 5  # search window for the IMU/OMC alignment offset

TOTAL_PARTICIPANTS = len(glob.glob("configs/P*"))
TOTAL_CONDITIONS = len({os.path.basename(p)[:-5] for p in glob.glob("configs/P*/*.yaml")
                         if not os.path.basename(p).startswith("ig_")})


def discover_trials():
    """(subject, trial) pairs with a completed (non-ig_*) result on disk."""
    pairs = []
    for path in sorted(glob.glob("result/P*/*.pkl")):
        subject = os.path.basename(os.path.dirname(path))
        trial = os.path.splitext(os.path.basename(path))[0]
        if trial.startswith("ig_"):
            continue
        pairs.append((subject, trial))
    return pairs


def load_result(subject, trial):
    with open(f"result/{subject}/{trial}.pkl", "rb") as f:
        (states_, globals_), info, settings = pickle.load(f)
    return states_, globals_, info, settings


def load_reference(subject, trial):
    df = pd.read_parquet(f"data/{subject}_{trial}_mean_FP.mat.parquet")
    dur = float(df[df["type"] == "dur"].iloc[0]["mean"][0])
    speed = float(df[df["type"] == "speed"].iloc[0]["mean"][0])
    angles, moments, grf = {}, {}, {}
    for name in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r"):
        row = df[(df["type"] == "angle") & (df["name"] == name)].iloc[0]
        angles[name] = np.asarray(row["mean"], dtype=float)  # (100,) rad, 0-100% cycle
        row = df[(df["type"] == "moment") & (df["name"] == name)].iloc[0]
        moments[name] = np.asarray(row["mean"], dtype=float)  # Nm
    for name in ("GRF_x_r", "GRF_y_r"):
        row = df[(df["type"] == "GRF") & (df["name"] == name)].iloc[0]
        grf[name] = 100.0 * np.asarray(row["mean"], dtype=float)  # BW fraction -> %BW
    return dur, speed, angles, moments, grf


def load_runtime_measurements():
    """{(subject, trial): {"wall_time_setup"/"wall_time_solve"/"wall_time_total": s}}.

    Populated by scripts/measure_runtime.py, which re-solves trials on an
    isolated worktree to record timing (see REPORT.md "Runtime"). Not every
    trial can necessarily be re-solved (e.g. warm-start files from an
    incompatible earlier model version), so this may be a subset of the
    discovered trials -- report.py handles that gracefully.
    """
    if not os.path.exists(RUNTIME_FILE):
        return {}
    with open(RUNTIME_FILE) as f:
        raw = json.load(f)
    return {tuple(k.split("/", 1)): v for k, v in raw.items()}


_coord_index_cache = {}


def get_coord_index(subject):
    """joint_key ("hip_r", ...) -> column index into a participant's q array.

    Loaded from that subject's own model (coordinates aren't guaranteed to be
    in the same order for every model file, and re-instantiating per subject
    also matches how these models are otherwise always used: freshly built
    per participant since ground-contact/actuator scaling is subject-specific).
    compile_eom=False skips the (slow, and unneeded here) symbolic EOM/JAX
    compile -- coordinate names are available without it.
    """
    if subject not in _coord_index_cache:
        from biosym.model.model import load_model

        model = load_model(f"models/participants/{subject}.yaml", compile_eom=False)
        names = list(model.coordinates.names)  # e.g. "q_hip_r", "q_knee_r", ...
        _coord_index_cache[subject] = {
            joint: names.index(f"q_{joint}") for joint in JOINTS
        }
    return _coord_index_cache[subject]


_participant_info_cache = None


def get_subject_height_mass(subject):
    """(height_m, mass_kg) for a subject, from data/ParticipantInfo.csv."""
    global _participant_info_cache
    if _participant_info_cache is None:
        _participant_info_cache = pd.read_csv("data/ParticipantInfo.csv").set_index("ID")
    row = _participant_info_cache.loc[subject]
    return float(row["BODYHEIGHT"]), float(row["BODYMASS"])


_full_model_cache = {}


def get_full_model(subject):
    """Fully-compiled model (symbolic EOM + JAX force functions) for a subject,
    needed to re-evaluate the contact/actuator force laws. Cached per subject
    since it's re-instantiated (not shared/global) precisely because ground
    contact and actuator scaling are subject-specific."""
    if subject not in _full_model_cache:
        from biosym.model.model import load_model

        _full_model_cache[subject] = load_model(f"models/participants/{subject}.yaml")
    return _full_model_cache[subject]


def reconstruct_grf_moments(subject, states_, globals_):
    """Re-evaluate the current contact/actuator force laws on a solved q/qd/
    actuator-state trajectory to recover GRF and joint moments (neither is
    part of the saved state -- see module docstring for why this can fail for
    older results).

    Returns a dict with keys grf_x_r/grf_y_r (%BW) and moment_hip_r/knee_r/
    ankle_r (Nm), each a (n_nodes+1,) array on the same grid as states_.q, plus
    "plausible" (bool): whether the reconstructed vertical GRF stayed
    non-negative within MIN_PLAUSIBLE_VERTICAL_GRF_PCTBW.
    """
    import jax
    import biosym.utils.states as sm

    model = get_full_model(subject)
    q = np.asarray(states_.q)
    qd = np.asarray(states_.qd)
    N = qd.shape[0]
    dur = float(globals_.dur)
    h = np.full(N - 1, dur / (N - 1))
    qdd = np.concatenate([(qd[0:1] - qd[-2:-1]) / h[-1], (qd[1:] - qd[:-1]) / h[:, None]], axis=0)

    batch = sm.States(
        q=q, qd=qd, qdd=qdd,
        tau=np.zeros((N, model.tau.n)),
        ext_forces=np.zeros((N, model.ext_forces.n)),
        ext_torques=np.zeros((N, model.ext_torques.n)),
        gc_model=np.zeros((N, model.default_states.gc_model.shape[0])),
        actuator_model=np.asarray(states_.actuator_model),
    )
    constants = model.default_constants

    ext_forces, _ext_torques = jax.vmap(model.run["gc_model"], in_axes=(0, None))(batch, constants)
    ext_forces = np.asarray(ext_forces).reshape(N, -1)
    tau = np.asarray(jax.vmap(model.run["actuator_model"], in_axes=(0, None))(batch, constants)).reshape(N, -1)

    mass_kg = float(model.default_constants.mass.sum())
    bw_n = mass_kg * 9.81
    grf_x_r = 100.0 * ext_forces[:, model.ext_forces.names.index("f_foot_r_x")] / bw_n
    grf_y_r = 100.0 * ext_forces[:, model.ext_forces.names.index("f_foot_r_y")] / bw_n
    moments = {joint: tau[:, model.tau.names.index(f"t_{joint}")] for joint in JOINTS}

    return {
        "grf_x_r": grf_x_r,
        "grf_y_r": grf_y_r,
        "moment_hip_r": moments["hip_r"],
        "moment_knee_r": moments["knee_r"],
        "moment_ankle_r": moments["ankle_r"],
        "plausible": bool(grf_y_r.min() >= MIN_PLAUSIBLE_VERTICAL_GRF_PCTBW),
    }


def best_offset_deg(sim_deg_100, ref_deg_100, max_offset):
    """Circularly shift ref to align with sim (both length-100, one gait cycle).
    Returns (best_offset_nodes, rmse_at_best, shifted_ref)."""
    best = None
    for off in range(-max_offset, max_offset + 1):
        shifted = np.roll(ref_deg_100, off)
        rmse = float(np.sqrt(np.mean((sim_deg_100 - shifted) ** 2)))
        if best is None or rmse < best[1]:
            best = (off, rmse, shifted)
    return best


def analyze_trials(trials):
    rows = []
    per_trial_curves = {}  # (subject, trial) -> dict of 100-node arrays for plotting
    runtimes = load_runtime_measurements()

    for subject, trial in trials:
        states_, globals_, info, settings = load_result(subject, trial)
        dur_full, speed_exp, angles_exp, moments_exp, grf_exp = load_reference(subject, trial)
        q_idx = get_coord_index(subject)

        q = np.asarray(states_.q)  # (nnodes+1, ncoords), 0..100% cycle (last node duplicates first)
        n_nodes = q.shape[0] - 1
        t_sim = np.linspace(0, 100, n_nodes + 1)  # percent gait cycle
        t_ref = np.linspace(0, 100, 101)[:-1]  # 100 experimental samples, 0..99%

        # Resample sim onto the same 100-sample grid as the reference (0..99% cycle).
        sim_100 = {joint: np.rad2deg(np.interp(t_ref, t_sim, q[:, idx])) for joint, idx in q_idx.items()}

        trial_result = {"subject": subject, "trial": trial}
        curves = {"t_pct": t_ref}
        offsets = {}
        for name, joint in (("hip_flexion_r", "hip_r"), ("knee_angle_r", "knee_r"), ("ankle_angle_r", "ankle_r")):
            ref_deg = np.rad2deg(angles_exp[name])
            off, rmse, shifted_ref = best_offset_deg(sim_100[joint], ref_deg, MAX_OFFSET_NODES)
            offsets[joint] = off
            trial_result[f"{joint}_rmse_deg"] = rmse
            curves[f"sim_{joint}"] = sim_100[joint]
            curves[f"ref_{joint}"] = shifted_ref

        trial_result["ja_rmse_deg_mean"] = np.mean([trial_result[f"{j}_rmse_deg"] for j in JOINTS])
        # Offset in ms, using the hip alignment (representative; offsets agree within
        # a node or two across joints in practice). The same offset is reused below
        # to align GRF/moments, rather than re-searching per signal.
        dt_ms = 1000.0 * dur_full / n_nodes
        hip_offset = offsets["hip_r"]
        trial_result["offset_nodes_hip"] = hip_offset
        trial_result["offset_ms_hip"] = hip_offset * dt_ms
        trial_result["offset_at_search_boundary"] = abs(hip_offset) == MAX_OFFSET_NODES

        # GRF/joint moments: reconstructed by re-evaluating the current force
        # laws (see module docstring) -- only trustworthy if physically plausible.
        try:
            recon = reconstruct_grf_moments(subject, states_, globals_)
            grf_moments_valid = recon["plausible"]
            grf_moments_error = None
        except Exception as e:  # noqa: BLE001 -- reconstruction failure is an expected, reportable outcome here
            recon, grf_moments_valid, grf_moments_error = None, False, str(e)

        trial_result["grf_moments_valid"] = grf_moments_valid
        trial_result["grf_moments_error"] = grf_moments_error
        trial_result["grf_rmse_pctbw"] = None
        trial_result["moment_rmse_pctbwbh"] = None
        if recon is not None:
            curves["sim_grf_x_r"] = np.interp(t_ref, t_sim, recon["grf_x_r"])
            curves["sim_grf_y_r"] = np.interp(t_ref, t_sim, recon["grf_y_r"])
            curves["ref_grf_x_r"] = np.roll(grf_exp["GRF_x_r"], hip_offset)
            curves["ref_grf_y_r"] = np.roll(grf_exp["GRF_y_r"], hip_offset)
            grf_rmses = [
                float(np.sqrt(np.mean((curves[f"sim_grf_{c}_r"] - curves[f"ref_grf_{c}_r"]) ** 2)))
                for c in ("x", "y")
            ]
            trial_result["grf_rmse_pctbw"] = float(np.mean(grf_rmses))

            moment_rmses_nm = []
            for name, joint in (("hip_flexion_r", "hip_r"), ("knee_angle_r", "knee_r"), ("ankle_angle_r", "ankle_r")):
                curves[f"sim_moment_{joint}"] = np.interp(t_ref, t_sim, recon[f"moment_{joint}"])
                curves[f"ref_moment_{joint}"] = np.roll(moments_exp[name], hip_offset)
                moment_rmses_nm.append(float(np.sqrt(np.mean(
                    (curves[f"sim_moment_{joint}"] - curves[f"ref_moment_{joint}"]) ** 2
                ))))
            height_m, mass_kg = get_subject_height_mass(subject)
            trial_result["moment_rmse_pctbwbh"] = 100.0 * np.mean(moment_rmses_nm) / (mass_kg * 9.81 * height_m)

        speed_sim = float(globals_.speed)
        trial_result["speed_sim_ms"] = speed_sim
        trial_result["speed_exp_ms"] = speed_exp
        trial_result["speed_error_ms"] = abs(speed_sim - speed_exp)

        trial_result["dur_s"] = dur_full
        trial_result["obj_val"] = info.get("obj_val")
        rt = runtimes.get((subject, trial), {})
        if "wall_time_total" in info:
            # This result was itself solved with the wall_time-recording patch
            # in place (collocation.py), so its own info dict has the timing.
            trial_result["wall_time_total_s"] = info["wall_time_total"]
            trial_result["wall_time_source"] = "solved_directly"
        elif "wall_time_total" in rt:
            # Older result predates the patch; timing came from a separate
            # offline re-solve recorded in result/runtime_measurements.json.
            trial_result["wall_time_total_s"] = rt["wall_time_total"]
            trial_result["wall_time_source"] = "offline_resolve"
        else:
            trial_result["wall_time_total_s"] = None
            trial_result["wall_time_source"] = None

        rows.append(trial_result)
        per_trial_curves[(subject, trial)] = curves

    return pd.DataFrame(rows), per_trial_curves


def readable_condition(condition):
    """"fastrunning" -> "Fast Running", "normwalking" -> "Normal Walking", etc.
    Falls back to a plain capitalized condition name for anything unexpected."""
    prefix_labels = {"fast": "Fast", "norm": "Normal", "slow": "Slow"}
    for suffix, activity in (("running", "Running"), ("walking", "Walking")):
        if condition.endswith(suffix):
            prefix = condition[: -len(suffix)]
            return f"{prefix_labels.get(prefix, prefix.capitalize())} {activity}"
    return condition.capitalize()


def _mean_sd_band(ax, t_pct, stack, color, label, ls="-", lw=2, alpha=0.15):
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    ax.plot(t_pct, mean, color=color, lw=lw, ls=ls, label=f"{label} (n={n})")
    if n > 1:
        sd = stack.std(axis=0)
        ax.fill_between(t_pct, mean - sd, mean + sd, color=color, alpha=alpha)
    return mean


def make_condition_figure(df, per_trial_curves, condition, fig_path):
    """3x3 grid for one trial condition (e.g. "normwalking"): hip/knee/ankle
    joint angle, hip/knee/ankle joint moment, GRF x/y, and a speed scatter --
    both simulated and reference (OMC) shown as mean +/- SD across the
    participants who have that condition. Moment/GRF panels only include
    trials whose reconstruction was flagged physically plausible (see
    reconstruct_grf_moments)."""
    sim_color, ref_color = "#1f77b4", "#444444"
    ja_keys = [(s, t) for s, t in per_trial_curves if t == condition]
    grfm_keys = [k for k in ja_keys if df.set_index(["subject", "trial"]).loc[k, "grf_moments_valid"]]
    t_pct = per_trial_curves[ja_keys[0]]["t_pct"]

    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle(f"{readable_condition(condition)}\nSimulated vs. reference (OMC), mean ± SD across participants",
                 fontsize=15, fontweight="bold")

    def joint_row(row_idx, kind, unit, keys):
        for col_idx, joint in enumerate(JOINTS):
            ax = axs[row_idx, col_idx]
            if keys:
                sim_stack = np.stack([per_trial_curves[k][f"sim_{kind}_{joint}"] if kind == "moment"
                                       else per_trial_curves[k][f"sim_{joint}"] for k in keys])
                ref_stack = np.stack([per_trial_curves[k][f"ref_{kind}_{joint}"] if kind == "moment"
                                       else per_trial_curves[k][f"ref_{joint}"] for k in keys])
                _mean_sd_band(ax, t_pct, ref_stack, ref_color, "Reference (OMC)", ls=":", lw=1.5)
                _mean_sd_band(ax, t_pct, sim_stack, sim_color, "Simulated")
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "no physically\nplausible reconstruction", ha="center", va="center",
                        fontsize=8, color="#888888", transform=ax.transAxes)
            ax.set_title(joint.replace("_r", "").capitalize() + f" {'flexion' if kind == 'angle' else 'moment'}")
            ax.set_xlabel("Gait cycle (%)")
            ax.set_ylabel(unit)
            ax.grid(alpha=0.3)

    joint_row(0, "angle", "Angle (deg)", ja_keys)
    joint_row(1, "moment", "Moment (Nm)", grfm_keys)

    for col_idx, (curve_key, title) in enumerate([("grf_x_r", "GRF fore-aft (x)"), ("grf_y_r", "GRF vertical (y)")]):
        ax = axs[2, col_idx]
        if grfm_keys:
            sim_stack = np.stack([per_trial_curves[k][f"sim_{curve_key}"] for k in grfm_keys])
            ref_stack = np.stack([per_trial_curves[k][f"ref_{curve_key}"] for k in grfm_keys])
            _mean_sd_band(ax, t_pct, ref_stack, ref_color, "Reference (OMC+FP)", ls=":", lw=1.5)
            _mean_sd_band(ax, t_pct, sim_stack, sim_color, "Simulated")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no physically\nplausible reconstruction", ha="center", va="center",
                    fontsize=8, color="#888888", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("GRF (%BW)")
        ax.grid(alpha=0.3)

    ax = axs[2, 2]
    sub = df[df["trial"] == condition]
    ax.scatter(sub["speed_exp_ms"], sub["speed_sim_ms"], color=sim_color, s=50,
               edgecolor="black", linewidth=0.5, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row["subject"], (row["speed_exp_ms"], row["speed_sim_ms"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    lo = min(sub["speed_exp_ms"].min(), sub["speed_sim_ms"].min(), 0)
    hi = max(sub["speed_exp_ms"].max(), sub["speed_sim_ms"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental speed (m/s)")
    ax.set_ylabel("Simulated speed (m/s)")
    ax.set_title("Gait speed")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return len(ja_keys), len(grfm_keys)


# Slowest to fastest, matching each condition's target speed range in
# Dorschky et al. (2019) Sec. 2.4.1 (walking: 0.9-1.0 / 1.2-1.4 / 1.8-2 m/s;
# running: 3.0-3.3 / 3.9-4.1 / 4.7-4.9 m/s) -- walking and running don't
# overlap, so this is speed order regardless of gait type.
TRIAL_SPEED_ORDER = [
    "slowwalking", "normwalking", "fastwalking",
    "slowrunning", "normrunning", "fastrunning",
]


def _speed_order_key(condition):
    try:
        return TRIAL_SPEED_ORDER.index(condition)
    except ValueError:
        return len(TRIAL_SPEED_ORDER)  # unknown conditions sort last, alphabetically among themselves


def make_figures(df, per_trial_curves, out_dir):
    """One 3x3 figure per trial condition. Returns
    [(condition, fig_path, n_participants_ja, n_participants_grf_moments), ...]."""
    results = []
    for condition in sorted(df["trial"].unique(), key=_speed_order_key):
        fig_path = os.path.join(out_dir, f"validation_{condition}.png")
        n_ja, n_grfm = make_condition_figure(df, per_trial_curves, condition, fig_path)
        results.append((condition, fig_path, n_ja, n_grfm))
    return results


def write_report(df, trials, figures):
    lines = []
    lines.append("# Dorschky et al. (2019) reproduction -- validation report\n")
    lines.append(
        "Accuracy and runtime of the completed muscle-driven IMU-tracking trials, "
        "against each subject's own optical-motion-capture (OMC) + force-plate reference "
        "data, and against the Dorschky et al. (2019) OCP baseline as compiled by "
        "SSPINNpose (Gambietz et al., *SSPINNpose: A Self-Supervised PINN for Inertial "
        "Pose and Dynamics Estimation*, ACM TIST, 2026).\n"
    )

    # Joint angle/speed/GRF/moment errors are aligned per trial by circularly
    # shifting the reference curve by whatever offset (searched in
    # +/-MAX_OFFSET_NODES nodes) minimizes the hip RMSE -- consistent with the
    # small systematic IMU/OMC synchronization lag (on the order of 2-3
    # samples / 20-30 ms at this data's node spacing) in Dorschky et al.'s data.
    # Per-trial numbers are in result/report_metrics.csv.
    # All four top-line metrics are pooled as RMSD (root-mean-square, not a
    # plain mean of absolute/RMS values): each per-trial number already is an
    # RMSD over that trial's own samples (100 gait-cycle points, or a single
    # speed difference), and since every trial contributes the same number of
    # samples, sqrt(mean(per-trial RMSD**2)) is the correct pooled RMSD across
    # trials/participants.
    def _rmsd(series):
        return float(np.sqrt(np.mean(np.square(series))))

    mean_ja = _rmsd(df["ja_rmse_deg_mean"])
    mean_speed_err = _rmsd(df["speed_error_ms"])
    valid_grfm = df[df["grf_moments_valid"]]
    mean_grf = _rmsd(valid_grfm["grf_rmse_pctbw"]) if len(valid_grfm) > 0 else None
    mean_moment = _rmsd(valid_grfm["moment_rmse_pctbwbh"]) if len(valid_grfm) > 0 else None
    runtime_trials = df[df["wall_time_total_s"].notna()]
    mean_rt = runtime_trials["wall_time_total_s"].mean() if len(runtime_trials) > 0 else None
    std_rt = runtime_trials["wall_time_total_s"].std() if len(runtime_trials) > 0 else None

    lines.append("## Comparison to Dorschky et al. (2019) / SSPINNpose baseline\n")
    lines.append(
        "| Metric | This reproduction | Dorschky (2019) |"
    )
    lines.append("|---|---|---|")
    lines.append(f"| Joint angle RMSD (deg) | {mean_ja:.2f} | {REFERENCE['JAE_deg']:.1f} |")
    lines.append(f"| Speed RMSD (m/s) | {mean_speed_err:.2f} | {REFERENCE['speed_error_ms']:.2f} |")
    if mean_grf is not None:
        lines.append(f"| GRF RMSD (%BW) | {mean_grf:.1f} | {REFERENCE['GRFE_pctBW']:.1f} |")
        lines.append(f"| Joint torque RMSD (%BWBH) | {mean_moment:.1f} | {REFERENCE['JTE_pctBWBH']:.1f} |")
    else:
        lines.append(f"| GRF RMSD (%BW) | N/A | {REFERENCE['GRFE_pctBW']:.1f} |")
        lines.append(f"| Joint torque RMSD (%BWBH) | N/A | {REFERENCE['JTE_pctBWBH']:.1f} |")
    ref_runtime_str = (
        f"({REFERENCE_RUNTIME_MEAN_MIN} +/- {REFERENCE_RUNTIME_SD_MIN} min, {REFERENCE_RUNTIME_HARDWARE})"
    )
    if mean_rt is not None:
        lines.append(f"| Runtime (min) | {mean_rt/60:.1f} +/- {std_rt/60:.1f} | {ref_runtime_str} |")
    else:
        lines.append(f"| Runtime (min) | N/A | {ref_runtime_str} |")
    lines.append("")

    lines.append("## Validation plots\n")
    lines.append(
        "One 3x3 grid per trial condition: hip/knee/ankle joint angle, hip/knee/ankle joint "
        "moment, fore-aft/vertical GRF (simulated mean, solid; reference/OMC or OMC+force-plate "
        "mean, dotted; shaded bands = +/-1 SD across participants for both simulated and "
        "reference curves, where more than one participant has that condition), and a speed "
        "scatter (experimental vs. simulated, one point per participant). Moment/GRF panels are "
        "reconstructed by re-evaluating the current contact/actuator force laws on the solved "
        "trajectory (they aren't part of the saved state) and are only shown where that "
        "reconstruction passed a plausibility check "
        f"(vertical GRF must stay above {MIN_PLAUSIBLE_VERTICAL_GRF_PCTBW:.0f}%BW). All summary "
        "metrics (this section and the table above) are RMSD.\n"
    )
    for condition, fig_path, n_ja, n_grfm in figures:
        lines.append(f"### {readable_condition(condition)}\n")
        lines.append(f"n={n_ja} for joint angle/speed, n={n_grfm} for moment/GRF.\n")
        lines.append(f"![{condition} validation grid]({fig_path})\n")

    with open("REPORT.md", "w") as f:
        f.write("\n".join(lines))


def main():
    trials = discover_trials()
    if not trials:
        raise SystemExit("No completed (non-ig_*) trials found under result/.")

    df, per_trial_curves = analyze_trials(trials)
    df.to_csv("result/report_metrics.csv", index=False)

    figures = make_figures(df, per_trial_curves, "result/report_figures")
    write_report(df, trials, figures)

    print("Wrote REPORT.md and result/report_metrics.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
