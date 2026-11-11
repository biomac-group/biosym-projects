"""
Figure 1 of the BiomechPriorVAE paper with the biosym solutions as "Ours".

    python scripts/plot_figure1.py [--results results/walking] [--out results/figure1_biosym.png]

Same data selection and styling as ``plot/figure01.py`` of the BiomechPriorVAE repository:

a) hip/knee/ankle angle and Fx/Fy ground reaction force over the gait cycle, walking at 1.2 m/s (solid) and running at
   3.5 m/s (dashed) overlaid. Reference with SD band, "Ours" as individual trials (thin) and trial mean with SD band,
   PredSim as mean with SD band; every cycle starts at its heel strike (``--align shift``: each condition circularly
   shifted, with one shift for all five channels, to its best alignment with the reference mean). Trials and
   PredSim runs are those within 0.05 m/s of the speed and of the gait class (walking; running or stiff running);
b) duty factor versus speed, one curve per source over all its set speeds, plus the "Ours" self-chosen speed (diamond);
c) cadence versus speed, the same, plus the literature reference relations.

Data: ``data/reference`` and ``data/baselines`` (``extract_paper_data.py``) and the trials
``<results>/v<speed>/trials.npz`` (set speeds) and ``<results>/free/trials.npz`` (self-chosen speed) of
``extract_biosym_trials.py``. Only converged runs without a second contact burst (double heel strike) are used.
"Ours" stores the duration of a step (half the gait cycle) like the paper's rows, which the contact detection and the
cadence use.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "figure_lib"))
import data_utils as du          # noqa: E402  (vendored from the BiomechPriorVAE repository)
import gait_contact as gc        # noqa: E402

COLORS = {"reference": "#000000", "ours": "#c9b437", "predsim": "#C56C14", "gaitdynamics": "#009E73",
          "gaitnet": "#C85293", "gaitencoder": "#C20707"}
LABELS = {"reference": "Reference", "ours": "Ours", "predsim": "PredSim", "gaitdynamics": "GaitDynamics",
          "gaitnet": "GGN", "gaitencoder": "GaitEncoder"}
MODES = (("walking", 1.2, "-", ("Walking",)),
         ("running", 3.5, (0, (2, 2)), ("Running", "Stiff Running")))
SPEED_TOL = 0.05
REFERENCE_SPEEDS = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5)
MAX_SPEED = 4.5
FS = 100
FONT = 6 * 16 / 7.087
LABEL_FONT = 8 * 16 / 7.087
JOINTS = ("Hip [deg]", "Knee [deg]", "Ankle [deg]")


def best_shift(sim, ref):
    """
    Circular shift [frames] of ``sim`` with the smallest mean absolute error to ``ref``.

    Each channel's error is divided by the peak-to-peak range of its reference, so the channels (degrees and body
    weights) enter equally.
    """
    scale = np.ptp(ref, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    errors = [np.nanmean(np.abs(np.roll(sim, s, axis=0) - ref) / scale) for s in range(ref.shape[0])]
    return int(np.argmin(errors))


ALIGN = "heelstrike"      # "heelstrike": every cycle starts at its heel strike; "shift": best circular shift to the reference


def shifted(trials, ref):
    """
    Trials [N, 100, C] aligned to ``ref`` [100, C], with their mean and SD.

    With ``ALIGN == "heelstrike"`` the cycles are left as they are: ``load_ours`` and ``load_predsim`` start every cycle
    at its heel strike (``heel_strike_start``), so the cycle bounds are the same physical event for every source and
    channel. With ``"shift"`` the trials are circularly shifted to the best (range-normalised mean absolute error)
    alignment of their mean with ``ref``, one shift for all C channels; that shift trades heel-strike timing against
    stance duration and can move the heel strike across the cycle bounds.

    Returns
    -------
    tuple
        Trials, mean and SD.
    """
    if ALIGN == "shift":
        trials = np.roll(trials, best_shift(trials.mean(axis=0), ref), axis=1)
    return trials, trials.mean(axis=0), trials.std(axis=0)


def robust_mean_sd(values):
    """Mean and SD after dropping points more than five scaled MADs from the median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    if mad > 0 and np.any(np.abs(values - median) <= 5 * mad):
        values = values[np.abs(values - median) <= 5 * mad]
    return float(values.mean()), float(values.std())


def curve(speeds, values, decimals=1):
    """Robust mean and SD of ``values`` at each rounded speed."""
    table = pd.DataFrame({"s": np.round(np.asarray(speeds, dtype=float), decimals), "v": values}).dropna()
    grouped = table.groupby("s")["v"].apply(lambda v: robust_mean_sd(v.values))
    return grouped.index.to_numpy(), np.array([m for m, _ in grouped]), np.array([s for _, s in grouped])


def duty_factor(vgrf, step):
    """Fraction of the cycle in stance; ``step`` is the stored duration of the trial."""
    return float(np.mean(gc.contact_mask(np.asarray(vgrf, dtype=float), FS / step if step > 0 else FS)))


def heel_strike_start(vgrf, threshold=0.05):
    """
    Frame to roll a cycle to so that its heel strike lands on frame 1, as the reference's does.

    The heel strike is the loading onset (vertical force crossing ``threshold`` [BW]) of the longest stance, so a
    spurious second touchdown does not move the cycle start. Cycles without a swing phase are left unrolled.
    """
    loaded = vgrf >= threshold
    onsets = np.flatnonzero(loaded & ~np.roll(loaded, 1))
    if len(onsets) == 0:
        return 0
    runs = [int(np.argmin(np.roll(loaded, -o))) or len(vgrf) for o in onsets]   # stance length from each onset
    return int(onsets[int(np.argmax(runs))]) - 1


def gait_class(grf_y, angles):
    """'Walking' when the vertical force is loaded at mid-cycle, else 'Running' or 'Stiff Running' (knee at contact)."""
    if grf_y[50] > 0.2:
        return "Walking"
    return "Stiff Running" if np.rad2deg(angles[0, 1]) > -10 else "Running"


def load_ours(path):
    """
    Trials of one ``trials.npz`` rolled to their heel strike, without those showing a second contact burst.

    Returns
    -------
    dict or None
        ``angles`` [N, 100, 3] rad, ``grf`` [N, 100, 2] BW, ``speed`` [m/s], ``step`` [s, half the gait cycle], ``gait``,
        ``seed``; ``None`` for a missing file. The kept and dropped seeds are printed.
    """
    if not Path(path).exists():
        return None
    t = np.load(path)
    keep, rows = [], {k: [] for k in ("angles", "grf", "speed", "step", "gait", "seed")}
    has_frontal = "frontal" in t.files
    if has_frontal:
        rows["frontal"] = []
    extras = [k for k in ("lumbar_ext", "objective") if k in t.files]
    for k in extras:
        rows[k] = []
    for i, (angles, grf, dur, speed, seed) in enumerate(zip(t["angles"], t["grf"], t["dur"], t["speed"], t["seed"])):
        step = dur / 2.0
        start = heel_strike_start(grf[:, 1])
        angles, grf = np.roll(angles, -start, axis=0), np.roll(grf, -start, axis=0)
        keep.append(gc.has_single_contact_phase(grf[:, 1], FS / step))
        for k, v in zip(("angles", "grf", "speed", "step", "gait", "seed"),
                        (angles, grf, speed, step, gait_class(grf[:, 1], angles), seed)):
            rows[k].append(v)
        if has_frontal:
            rows["frontal"].append(np.roll(t["frontal"][i], -start, axis=0))
        if "lumbar_ext" in extras:
            rows["lumbar_ext"].append(np.roll(t["lumbar_ext"][i], -start, axis=0))
        if "objective" in extras:
            rows["objective"].append(t["objective"][i])
    keep = np.array(keep, dtype=bool)
    on_bound = t["clamped"] if "clamped" in t.files else np.zeros(len(keep), dtype=bool)
    double = np.array(rows["seed"])[~keep]
    clamped = np.array(rows["seed"])[on_bound & keep]
    keep &= ~on_bound            # a speed or half-cycle duration on its bound is not a genuine optimum
    out = {k: np.array(v)[keep] for k, v in rows.items()}
    print(f"{Path(path).parent.name}: kept seeds {list(out['seed'])}, dropped: double heel strike {list(double)}, "
          f"on a bound {list(clamped)}")
    return out


def concat(parts):
    """Concatenate the dictionaries of ``load_ours``."""
    parts = [p for p in parts if p is not None and len(p["seed"])]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


def load_predsim():
    """PredSim ensemble runs that have a single contact phase."""
    z = np.load(PROJECT / "data/baselines/predsim_ensemble.npz", allow_pickle=True)
    keep = z["single_contact"]
    out = {k: z[k][keep] for k in ("angles", "grf", "speed", "dur", "duty", "gait")}
    starts = [heel_strike_start(g[:, 1]) for g in out["grf"]]            # every run starts at its heel strike
    out["angles"] = np.array([np.roll(a, -s, axis=0) for a, s in zip(out["angles"], starts)])
    out["grf"] = np.array([np.roll(g, -s, axis=0) for g, s in zip(out["grf"], starts)])
    return out


def reference_duty():
    """Duty factor of the reference GRF mean and of its +/- 1 SD envelope."""
    speeds, means, sds = [], [], []
    for speed in REFERENCE_SPEEDS:
        ref = du.get_reference_data(speed, root_path=str(PROJECT / "data/reference"))
        if ref is None or "Fy" not in ref:
            continue
        fy, sd = np.asarray(ref["Fy"]["mean"]), np.asarray(ref["Fy"]["std"])
        if np.nanmax(np.abs(fy)) > 5:
            fy, sd = fy / 9.81, sd / 9.81
        mean = float(np.mean(gc.contact_mask(fy, FS)))
        lo, hi = (float(np.mean(gc.contact_mask(fy + s * sd, FS))) for s in (-1, 1))
        speeds.append(speed); means.append(mean); sds.append(max(abs(mean - lo), abs(mean - hi)))
    return np.array(speeds), np.array(means), np.array(sds)


def draw_curve(ax, source, speeds, means, sds):
    keep = speeds <= MAX_SPEED
    speeds, means, sds = speeds[keep], means[keep], sds[keep]
    if len(speeds) == 0:
        return
    if source == "reference":
        ax.plot(speeds, means, color=COLORS[source], lw=2, zorder=1)
        ax.fill_between(speeds, means - sds, means + sds, color=COLORS[source], alpha=0.15, lw=0, zorder=1)
    else:
        ax.plot(speeds, means, color=COLORS[source], ls=":", lw=1.5, zorder=2)
        ax.errorbar(speeds, means, yerr=sds, color=COLORS[source], fmt="none", elinewidth=1, capsize=2, alpha=0.6,
                    zorder=2)


def draw_self_chosen(ax, speed, values):
    """Diamond at the robust mean speed and value of the self-chosen runs, with SD bars in both directions."""
    speed_mean, speed_sd = robust_mean_sd(speed)
    mean, sd = robust_mean_sd(values)
    if np.isfinite(mean):
        ax.errorbar([speed_mean], [mean], xerr=[speed_sd], yerr=[sd], color=COLORS["ours"], marker="D", markersize=6,
                    markeredgecolor="k", markeredgewidth=0.6, capsize=3, linestyle="none", zorder=5)


def panel_a(fig, spec, ours, predsim):
    """Walking and running overlaid: hip/knee/ankle in the left column, Fx/Fy in the right, legend in the spare cell."""
    inner = GridSpecFromSubplotSpec(3, 2, subplot_spec=spec, hspace=0.3, wspace=0.3)
    axes_angle = [fig.add_subplot(inner[i, 0]) for i in range(3)]
    axes_grf = [fig.add_subplot(inner[i, 1]) for i in range(2)]
    x = np.arange(100)

    def draw_trials(ax, result, color, ls, thin):
        trials, mean, sd = result
        if thin:
            ax.plot(trials.T, color=color, alpha=0.35, ls=ls, lw=0.8, zorder=2)
        ax.plot(mean, color=color, lw=2 if thin else 1.5, ls=ls, zorder=3)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12 if thin else 0.08, lw=0, zorder=2)

    for _, speed, ls, classes in MODES:
        ref = du.get_reference_data(speed, root_path=str(PROJECT / "data/reference"))
        ref_mean = np.array([ref["hip"]["mean"], ref["knee"]["mean"], ref["ankle"]["mean"],
                             ref["Fx"]["mean"], ref["Fy"]["mean"]]).T
        ref_sd = np.array([ref["hip"]["std"], ref["knee"]["std"], ref["ankle"]["std"],
                           ref["Fx"]["std"], ref["Fy"]["std"]]).T
        mine = (np.isclose(ours["speed"], speed, atol=SPEED_TOL) & np.isin(ours["gait"], classes)
                if ours else np.zeros(0, dtype=bool))
        theirs = np.isclose(predsim["speed"], speed, atol=SPEED_TOL) & np.isin(predsim["gait"], classes)
        # one circular shift for all five channels (hip, knee, ankle, Fx, Fy), so a trial stays consistent
        for source, selected, thin in (("ours", mine, True), ("predsim", theirs, False)):
            data = ours if source == "ours" else predsim
            if not selected.any():
                continue
            trials = np.concatenate([np.rad2deg(data["angles"][selected]), data["grf"][selected]], axis=2)
            aligned = shifted(trials, ref_mean)
            for i, ax in enumerate(axes_angle + axes_grf):
                draw_trials(ax, tuple(r[..., i] for r in aligned), COLORS[source], ls, thin)
        for i, ax in enumerate(axes_angle + axes_grf):
            ax.plot(ref_mean[:, i], color=COLORS["reference"], ls=ls, lw=2, zorder=1)
            ax.fill_between(x, ref_mean[:, i] - ref_sd[:, i], ref_mean[:, i] + ref_sd[:, i],
                            color=COLORS["reference"], alpha=0.08, lw=0, zorder=1)

    for ax, label in zip(axes_angle + axes_grf, list(JOINTS) + ["Fx [BW]", "Fy [BW]"]):
        ax.set_ylabel(label)
        ax.set_xlim(0, 100)
    axes_angle[1].set_yticks([-90, -45, 0])
    axes_grf[0].set_yticks([-0.4, -0.2, 0, 0.2])
    axes_grf[0].set_ylim(-0.5, 0.3)
    axes_angle[-1].set_xlabel("Gait cycle [%]")
    axes_grf[-1].set_xlabel("Gait cycle [%]")
    for ax in axes_angle[:-1] + axes_grf[:-1]:
        ax.tick_params(labelbottom=False)
    legend = fig.add_subplot(inner[2, 1])
    legend.axis("off")
    legend.legend(handles=[Line2D([0], [0], color=COLORS[s], lw=2, label=LABELS[s])
                           for s in ("reference", "ours", "predsim", "gaitdynamics", "gaitnet", "gaitencoder")],
                  loc=(-0.1, 0.1), fontsize=FONT, frameon=False, ncol=2, handlelength=1.2)
    axes_angle[0].text(-0.22, 1.08, "a", transform=axes_angle[0].transAxes, fontsize=LABEL_FONT, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(PROJECT / "results/walking"))
    parser.add_argument("--out", default=str(PROJECT / "results/figure1_biosym.png"))
    parser.add_argument("--align", choices=("heelstrike", "shift"), default="heelstrike",
                        help="cycles start at the heel strike (default), or are shifted to the best fit with the reference")
    args = parser.parse_args()
    global ALIGN
    ALIGN = args.align
    results = Path(args.results)

    ours = concat([load_ours(f) for f in sorted(results.glob("v*/trials.npz"))])
    free = load_ours(results / "free" / "trials.npz")
    predsim = load_predsim()
    base = PROJECT / "data/baselines"
    sets = {name: pd.read_csv(base / f"{name}_setspeed.csv") for name in ("gaitdynamics", "gaitnet", "gaitencoder")}
    ours_duty = lambda d: [duty_factor(g[:, 1], s) for g, s in zip(d["grf"], d["step"])]       # noqa: E731
    ours_cadence = lambda d: 1.0 / d["step"]                                                    # noqa: E731

    with plt.rc_context({"font.size": FONT}):
        fig = plt.figure(figsize=(16, 8.5))
        top = GridSpec(1, 2, width_ratios=[0.6, 0.4], wspace=0.3)
        panel_a(fig, top[0, 0], ours, predsim)
        right = GridSpecFromSubplotSpec(2, 1, subplot_spec=top[0, 1], hspace=0.55)

        ax = fig.add_subplot(right[0])
        draw_curve(ax, "predsim", *curve(predsim["speed"], predsim["duty"]))
        for name in ("gaitdynamics", "gaitnet"):
            draw_curve(ax, name, *curve(sets[name]["speed"].values, sets[name]["duty_factor"].values))
        if ours:
            draw_curve(ax, "ours", *curve(ours["speed"], ours_duty(ours)))
        draw_curve(ax, "reference", *reference_duty())
        if free:
            draw_self_chosen(ax, free["speed"], ours_duty(free))
        ax.set_xlabel("Speed [m/s]"); ax.set_ylabel("Duty factor")
        ax.text(-0.15, 1.05, "b", transform=ax.transAxes, fontsize=LABEL_FONT, fontweight="bold")

        ax = fig.add_subplot(right[1])
        walk = np.arange(0.7, 1.8, 0.01)
        ax.plot(walk, (0.27 * walk + 0.57) * 2, color=COLORS["reference"], lw=1.5, zorder=1)
        ax.plot([2.68, 2.82, 2.98, 3.16, 3.35, 3.58, 3.83],
                np.array([169.2, 169.9, 170.5, 172.3, 174.3, 175.9, 177.9]) / 60, color=COLORS["reference"], lw=1.5)
        draw_curve(ax, "predsim", *curve(predsim["speed"], 2.0 / predsim["dur"]))
        for name in ("gaitdynamics", "gaitnet", "gaitencoder"):
            draw_curve(ax, name, *curve(sets[name]["speed"].values, sets[name]["cadence_steps_per_s"].values))
        if ours:
            draw_curve(ax, "ours", *curve(ours["speed"], ours_cadence(ours)))
        if free:
            draw_self_chosen(ax, free["speed"], ours_cadence(free))
        ax.set_xlabel("Speed [m/s]"); ax.set_ylabel("Cadence [steps/s]")
        ax.text(-0.15, 1.05, "c", transform=ax.transAxes, fontsize=LABEL_FONT, fontweight="bold")

        for a in fig.axes:
            if a.axison:
                a.spines[["top", "right"]].set_visible(False)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
