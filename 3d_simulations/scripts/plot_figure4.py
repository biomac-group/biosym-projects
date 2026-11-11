"""
Figure 4 of the BiomechPriorVAE paper (the ``figure04b.py`` variant) with the biosym solutions: the state-prior OCP as
an engineering design tool.

    python scripts/plot_figure4.py [--results results] [--out results/figure4_biosym.png]

Geometry, fonts, colours and reference data follow ``plot/figure04a.py`` / ``plot/figure04b.py`` of the BiomechPriorVAE
repository (16 x 10 in drawn, fonts inflated for the printed width; top row a | GRF stack | b, bottom row c):

a) Umberger-scored metabolic cost [J/kg/m] versus speed (0.8-4.0 m/s, the paper's range), Bhargava cost of transport
   ("Metabolic", ``results/walking``) against the cubed muscle effort ("Effort", ``results/effort``): mean +/- SD band and
   the speed of the minimum cost; behind them in black the literature: Ralston (1958, solid), Abe et al. (2019, dashed,
   walking and running with SD) and Vernillo (dotted);
   (unlettered) vertical (solid) and fore-aft (dashed) ground reaction force at 1.6 m/s for both objectives against the
   reference, and the peak vertical force at 2.5, 3.5 and 4.5 m/s against the reference band;
b) bilateral hip-abductor weakness, free-speed runs (``results/weakfree<remaining %>/free``, 100 % is ``results/walking/free``): peak
   pelvic drop (pelvis obliquity, solid) and trunk lean (lumbar bending, dashed) against the remaining strength with the
   4-5 deg clinical band, and the pelvis rotation over the gait cycle coloured by the weakness applied;
c) ground contact in running at 4.5 m/s with the effort objective: the nominal contact (``results/effort``), four times
   the stiffness (``results/stiff``) and no dissipation (``results/nodamp``): peak vertical force against the reference
   and the Umberger cost; the mesh render of the paper is not drawn.

Only converged runs without double heel strikes, not on a speed or duration bound and within mean + 2 SD of the
Bhargava cost of their speed group (the notebook's filter) are used; whatever is not (yet) solved is left blank and
its count is printed.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_figure1 as f1          # noqa: E402  (trial loading, heel-strike roll, screens, robust statistics)

REFERENCE = "#000000"
COST_COLORS = {"effort": "#D55E00", "metabolic": "#0072B2"}
COST_SHORT_LABEL = {"effort": "Effort", "metabolic": "Metabolic"}
WEAKNESS_COLORS = {"drop": "#CC79A7", "lean": "#009E73"}
WEAKNESS_LABELS = {"drop": "Pelvic drop", "lean": "Trunk lean"}
ANGLE_LINESTYLE = {"drop": "-", "lean": "--"}
FOOTWEAR = {"smoothsphere": ("effort", COST_COLORS["effort"], "Nominal"), "stiffer4x": ("stiff", "#8B6FA8", "Stiff"),
            "nodamp": ("nodamp", "#D62728", "Bouncy")}
WEAKNESS_CMAP = plt.get_cmap("PRGn")
LITERATURE_LINESTYLE = {"ralston": "-", "abe_walk": "--", "abe_run": "--", "vernillo": ":"}
PEAK_VGRF_SPEEDS = (2.5, 3.5, 4.5)
GRF_SHAPE_SPEED = 1.6
MIN_TRIALS_PER_SPEED = 3                # panel a: the robust mean/SD cannot trim outliers from fewer solves (not in the paper)
COST_SWEEP_BOUNDS = (0.8, 4.0)          # m/s; the paper's 0.83-4.0, from our lowest speed
# Panel b follows figure04b.py of the paper: free-speed runs (results/weakfree<remaining %>/free), the unweakened comparator
# is the ordinary free-speed set (results/walking/free), and every level passes the same solve-quality filters.
BASELINE_FOLDER = ("walking", "free")
MIN_FIRST_VGRF_PEAK_BW, FIRST_PEAK_PROMINENCE_BW = 0.8, 0.02   # loading-response peak of the vertical force
MIN_LUMBAR_EXTENSION_DEG = -30.0                                 # cycle-mean lumbar extension, one-sided
MAX_OBJECTIVE_GAP = 0.10                                         # to the best solve of the level
DROP_THRESHOLD, LEAN_THRESHOLD = 4.0, 5.0            # deg, clinical thresholds of the paper
FONT = f1.FONT
LABEL_FONT = f1.LABEL_FONT
_R = 75
MIN_COST_ANNOTATION_OFFSET = {
    "metabolic": (_R * math.cos(math.radians(15)), -_R * math.sin(math.radians(15))),
    "effort": (-_R * math.sin(math.radians(15)), _R * math.cos(math.radians(15))),
}


def panel_label(fig, letter, x_ax, y_ax=None, x_offset=-0.28, y_offset=1.08):
    """Bold panel letter in figure coordinates, x from ``x_ax`` and y from ``y_ax``."""
    y_ax = y_ax if y_ax is not None else x_ax
    pos_x, pos_y = x_ax.get_position(), y_ax.get_position()
    fig.text(pos_x.x0 + x_offset * (pos_x.x1 - pos_x.x0), pos_y.y0 + y_offset * (pos_y.y1 - pos_y.y0), letter,
             fontsize=LABEL_FONT, fontweight="bold")


def swept_speeds(results, variant):
    """Set speeds [m/s] that have a ``trials.npz`` in ``results/<variant>/v<speed>``; the panels plot whatever was swept."""
    speeds = []
    for folder in Path(results, variant).glob("v*/trials.npz"):
        try:
            speeds.append(round(float(folder.parent.name[1:]), 2))
        except ValueError:
            pass
    return sorted(speeds)


def load(results, variant, speed):
    """Screened trials of one variant at one speed, with their metabolic costs; ``None`` without converged runs."""
    folder = Path(results) / variant / f"v{speed:.2f}"
    ours = f1.load_ours(folder / "trials.npz")
    if ours is None or len(ours["seed"]) == 0:
        return None
    path = folder / "metabolics.csv"
    if path.exists():
        cost = pd.read_csv(path).set_index("seed")
        # the notebook's cost bound: the Bhargava cost of a run within mean + 2 SD of its own speed group (taken over
        # all converged runs, as in data_exploration_main.ipynb)
        upper = cost["bhargava"].mean() + 2 * cost["bhargava"].std()
        within = np.array([cost["bhargava"].get(int(s), np.inf) <= upper for s in ours["seed"]]) \
            if len(cost) > 1 else np.zeros(len(ours["seed"]), dtype=bool)
        print(f"{variant} {speed}: cost bound {upper:.2f} J/kg/m drops seeds {list(ours['seed'][~within])}")
        ours = {k: v[within] for k, v in ours.items()}
        if len(ours["seed"]) == 0:
            return None
        ours["umberger"] = np.array([cost["umberger"].get(int(s), np.nan) for s in ours["seed"]])
    return ours


def peak_vgrf(ours):
    return ours["grf"][..., 1].max(axis=1)


def reference(speed):
    return f1.du.get_reference_data(speed, root_path=str(f1.PROJECT / "data/reference"))


def reference_peak(speed):
    """Mean and SD of the reference vertical-force curve at its own peak."""
    ref = reference(speed)
    mean, sd = np.asarray(ref["Fy"]["mean"], dtype=float), np.asarray(ref["Fy"]["std"], dtype=float)
    i = int(np.argmax(mean))
    return float(mean[i]), float(sd[i])


def strip_plot(ax, groups, colors, labels, y_label, jitter=0.06):
    """One jittered strip per group with the robust mean +/- SD (diamond)."""
    rng = np.random.default_rng(0)
    for x, key in enumerate(groups):
        values = np.asarray(groups[key], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        ax.scatter(x + rng.uniform(-jitter, jitter, size=len(values)), values, color=colors[key], alpha=0.55, s=14,
                   linewidth=0)
        mean, sd = f1.robust_mean_sd(values)
        ax.errorbar([x], [mean], yerr=[sd], color=colors[key], marker="D", markersize=6, markeredgecolor="k",
                    markeredgewidth=0.6, capsize=3, linestyle="none", zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels[k] for k in groups])
    ax.set_ylabel(y_label)


def literature_cost():
    """(speeds, cost J/kg/m, SD or None) of Ralston (1958), Abe et al. (2019) and Vernillo's quadratic."""
    out = {}
    v = np.linspace(24.4, 97.6, 50)                                         # m/min, Ralston's tested range
    out["ralston"] = (v / 60, (29 + 0.0053 * v ** 2) / v * 4.184, None)
    for key, name in (("abe_walk", "peerj-07-8290-s002.csv"), ("abe_run", "peerj-07-8290-s003.csv")):
        table = pd.read_csv(PROJECT / "data/reference" / name)
        columns = [c for c in table.columns if c != "Participant No."]
        out[key] = (np.array([float(c.split()[0]) for c in columns]), table[columns].mean(axis=0).to_numpy(),
                    table[columns].std(axis=0).to_numpy())
    v = np.linspace(1.78, 5.14, 60)
    out["vernillo"] = (v, ((0.006 * v ** 2 + 0.027 * v + 0.048) * 4184 / 60) / v, None)
    return out


def panel_cost(fig, spec, data):
    ax = fig.add_subplot(spec)
    for key, linestyle in LITERATURE_LINESTYLE.items():
        speeds, cost, sds = literature_cost()[key]
        ax.plot(speeds, cost, color="k", linewidth=1.8, linestyle=linestyle, zorder=1)
        if sds is not None:
            ax.fill_between(speeds, cost - sds, cost + sds, color="k", alpha=0.12, linewidth=0, zorder=0)
    for cost in ("metabolic", "effort"):
        speeds, means, sds = [], [], []
        for s in sorted(data[cost]):
            d = data[cost].get(s)
            if COST_SWEEP_BOUNDS[0] - 1e-6 <= s <= COST_SWEEP_BOUNDS[1] + 1e-6 and d is not None and "umberger" in d \
                    and len(d["umberger"]) >= MIN_TRIALS_PER_SPEED:
                mean, sd = f1.robust_mean_sd(d["umberger"])
                if np.isfinite(mean):
                    speeds.append(s); means.append(mean); sds.append(sd)
        if not speeds:
            continue
        speeds, means, sds = map(np.array, (speeds, means, sds))
        color = COST_COLORS[cost]
        ax.plot(speeds, means, color=color, linewidth=2, label=COST_SHORT_LABEL[cost], zorder=3)
        ax.fill_between(speeds, means - sds, means + sds, color=color, alpha=0.15, linewidth=0, zorder=2)
        i = int(np.nanargmin(means))
        ax.plot([speeds[i]], [means[i]], marker="o", color=color, markersize=6, markeredgecolor="k",
                markeredgewidth=0.8, zorder=5)
        ax.annotate(f"{speeds[i]:.2f} m/s", xy=(speeds[i], means[i]), xytext=MIN_COST_ANNOTATION_OFFSET[cost],
                    textcoords="offset points", color=color, fontsize=FONT, ha="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
    ax.set_xlim(*COST_SWEEP_BOUNDS)
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Metabolic cost [J/kg/m]")
    panel_label(fig, "a", ax)
    return ax


def draw_condition_curves(ax, curves, colors, reference_curve=None, linestyle="-"):
    """Mean +/- SD curve per condition, circularly shifted to the reference, with the reference behind."""
    ref_mean = None
    if reference_curve is not None:
        ref_mean, ref_std = reference_curve
        ax.plot(ref_mean, color=REFERENCE, linewidth=2, linestyle=linestyle, zorder=1)
        ax.fill_between(range(len(ref_mean)), ref_mean - ref_std, ref_mean + ref_std, color=REFERENCE, alpha=0.1,
                        linewidth=0, zorder=1)
    for key, trials in curves.items():
        if len(trials) == 0:
            continue
        trials, mean, std = f1.shifted(trials[..., None], ref_mean[:, None])
        trials, mean, std = trials[..., 0], mean[:, 0], std[:, 0]
        ax.plot(mean, color=colors[key], linewidth=2, linestyle=linestyle, zorder=3)
        ax.fill_between(range(len(mean)), mean - std, mean + std, color=colors[key], alpha=0.15, linewidth=0, zorder=2)


def panel_grf(fig, spec, data):
    """Stance shape at 1.6 m/s and the peak vertical force at the running speeds, effort against metabolic."""
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=0.6)
    ax_grf, ax_peak = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    ref = reference(GRF_SHAPE_SPEED)
    for column, linestyle, key in ((1, "-", "Fy"), (0, "--", "Fx")):
        curves = {cost: (data[cost][GRF_SHAPE_SPEED]["grf"][..., column][data[cost][GRF_SHAPE_SPEED]["gait"] == "Walking"]
                         if data[cost].get(GRF_SHAPE_SPEED) is not None else np.zeros((0, 100)))
                  for cost in ("metabolic", "effort")}
        draw_condition_curves(ax_grf, curves, COST_COLORS, (np.asarray(ref[key]["mean"]), np.asarray(ref[key]["std"])),
                              linestyle)
    ax_grf.set_xlim(0, 100)
    ax_grf.set_xlabel("Gait cycle [%]")
    ax_grf.set_ylabel("GRF [BW]")
    ax_grf.text(0.98, 0.95, "solid: vertical\ndashed: fore-aft", transform=ax_grf.transAxes, fontsize=FONT * 0.8,
                color="0.3", va="top", ha="right")

    rng = np.random.default_rng(0)
    group_width = 0.32
    for i, speed in enumerate(PEAK_VGRF_SPEEDS):
        ref_mean, ref_std = reference_peak(speed)
        ax_peak.fill_between([i - 0.4, i + 0.4], ref_mean - ref_std, ref_mean + ref_std, color=REFERENCE, alpha=0.1,
                             linewidth=0, zorder=0)
        ax_peak.hlines(ref_mean, i - 0.4, i + 0.4, color=REFERENCE, linestyle="--", linewidth=1.5, zorder=1)
        for cost, offset in (("effort", -group_width / 2), ("metabolic", group_width / 2)):
            d = data[cost].get(speed)
            if d is None:
                continue
            values = peak_vgrf(d[np.isin(d["gait"], ("Running", "Stiff Running"))] if False else d)
            x0 = i + offset
            ax_peak.scatter(x0 + rng.uniform(-group_width * 0.18, group_width * 0.18, size=len(values)), values,
                            color=COST_COLORS[cost], alpha=0.55, s=14, linewidth=0, zorder=3)
            mean, sd = f1.robust_mean_sd(values)
            ax_peak.errorbar([x0], [mean], yerr=[sd], color=COST_COLORS[cost], marker="D", markersize=6,
                             markeredgecolor="k", markeredgewidth=0.6, capsize=3, linestyle="none", zorder=5)
    ax_peak.set_xticks(range(len(PEAK_VGRF_SPEEDS)))
    ax_peak.set_xticklabels([f"{s:g}" for s in PEAK_VGRF_SPEEDS])
    ax_peak.set_xlabel("Speed [m/s]")
    ax_peak.set_ylabel("Peak vertical GRF [BW]")


def first_vgrf_peak(vgrf):
    """Loading-response peak [BW]: the first local maximum (prominence 0.02 BW) of the heel-strike-first vertical force."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(vgrf, prominence=FIRST_PEAK_PROMINENCE_BW)
    return float(vgrf[peaks[0]]) if len(peaks) else float(np.max(vgrf))


def solve_filter(d, level):
    """
    Solves of one weakness level that are a usable walking gait, as ``solve_rejections`` of figure04b.py.

    A solve is dropped for a first vertical-force peak below 0.8 BW, for not being walking, for a cycle-mean lumbar
    extension below -30 deg, and, among the survivors, for an objective more than 0.10 above the level's best (a worse
    local optimum). Solves on a speed or duration bound were already removed by ``load_ours``.
    """
    if d is None or len(d["seed"]) == 0:
        return d
    reasons = []
    for i in range(len(d["seed"])):
        peak1 = first_vgrf_peak(d["grf"][i, :, 1])
        lumbar = float(np.rad2deg(d["lumbar_ext"][i].mean()))
        if peak1 < MIN_FIRST_VGRF_PEAK_BW:
            reasons.append(f"first vGRF peak {peak1:.2f} BW")
        elif d["gait"][i] != "Walking":
            reasons.append(f"{str(d['gait'][i]).lower()} solve")
        elif lumbar < MIN_LUMBAR_EXTENSION_DEG:
            reasons.append(f"lumbar ext {lumbar:.1f} deg")
        else:
            reasons.append(None)
    survivors = [i for i, r in enumerate(reasons) if r is None]
    if survivors:
        best = min(d["objective"][i] for i in survivors)
        for i in survivors:
            if d["objective"][i] - best > MAX_OBJECTIVE_GAP:
                reasons[i] = f"objective {d['objective'][i] - best:+.3f} vs level best"
    dropped = [r for r in reasons if r is not None]
    if dropped:
        print(f"weakness {level}% remaining: dropped {len(dropped)} solve(s) -- {', '.join(dropped)}")
    keep = np.array([r is None for r in reasons], dtype=bool)
    return {k: v[keep] for k, v in d.items()}


def weakness_levels(results):
    """{remaining strength [%]: filtered free-speed trials}; 100 % is the ordinary free-speed set."""
    folders = {100: Path(results, *BASELINE_FOLDER)}
    for folder in Path(results).glob("weakfree*/free"):
        folders[int(folder.parent.name.removeprefix("weakfree"))] = folder
    levels = {}
    for level, folder in sorted(folders.items(), reverse=True):
        d = f1.load_ours(folder / "trials.npz")
        if d is not None and "lumbar_ext" in d and "objective" in d:
            d = solve_filter(d, level)
        else:
            d = None
        if d is not None and len(d["seed"]) and "frontal" in d:
            levels[level] = d
    return levels


def panel_weakness(fig, spec, results, ax_a):
    """Peak pelvic drop and trunk lean against the remaining strength, and the pelvis rotation traces (free speed)."""
    rows = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[1, 0.7], hspace=0.75)
    top, bottom = fig.add_subplot(rows[0]), fig.add_subplot(rows[1])
    levels = weakness_levels(results)
    for level, d in levels.items():
        print(f"weakness {level}% remaining: {len(d['seed'])} trials, speeds {np.round(d['speed'], 2)}")

    top.axhspan(DROP_THRESHOLD, LEAN_THRESHOLD, color=REFERENCE, alpha=0.12, linewidth=0, zorder=0)
    for key, column in (("drop", 0), ("lean", 2)):
        xs = np.array(sorted(levels), dtype=float)
        stats = [f1.robust_mean_sd(np.rad2deg(np.abs(levels[int(p)]["frontal"][..., column]).max(axis=1))) for p in xs]
        if len(xs):
            top.errorbar(xs, [m for m, _ in stats], yerr=[s for _, s in stats], color=WEAKNESS_COLORS[key],
                         linestyle=ANGLE_LINESTYLE[key], marker="o", markersize=4, linewidth=1.6, capsize=3,
                         label=WEAKNESS_LABELS[key], zorder=3)
    if len(levels) > 1:
        top.set_ylim(bottom=0)
        top.legend(fontsize=FONT * 0.85, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
                   handlelength=1.6, columnspacing=1.2, borderpad=0)
    else:
        top.set_ylim(0, 8)
        top.text(0.5, 0.25, "weakness sweep not yet available", transform=top.transAxes, ha="center", va="center",
                 fontsize=FONT, color=REFERENCE, alpha=0.5)
    top.set_xlim(105, 0)
    top.set_xlabel("Remaining abductor strength [%]")
    top.set_ylabel("Peak angle [$\\degree$]")

    norm = plt.Normalize(0, 140)
    for p in sorted(levels):
        curve = np.rad2deg(levels[p]["frontal"][..., 1]).mean(axis=0)
        bottom.plot(np.linspace(0, 100, len(curve), endpoint=False), curve, color=WEAKNESS_CMAP(norm(100 - p)),
                    linewidth=1.4, zorder=3)
    if levels:
        sm = plt.cm.ScalarMappable(cmap=WEAKNESS_CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=bottom, pad=0.08, fraction=0.06, aspect=12)
        cbar.ax.set_ylim(0, 100)
        cbar.set_label("Weakness applied [%]", fontsize=FONT * 0.85)
        cbar.ax.tick_params(labelsize=FONT * 0.75)
        bottom.axhline(0, color="0.7", linewidth=0.8, zorder=0)
    else:
        bottom.set_ylim(-15, 15)
    bottom.set_xlim(0, 100)
    bottom.set_xlabel("Gait cycle [%]")
    bottom.set_ylabel("Pelvis rotation [$\\degree$]")
    panel_label(fig, "b", top, y_ax=ax_a)
    return {p: len(d["seed"]) for p, d in levels.items()}


def panel_footwear(fig, spec, results, ax_a):
    outer = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[0.44, 0.56], wspace=0.2)
    strips = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.75)
    ax_vgrf, ax_umberger, ax_render = fig.add_subplot(strips[0]), fig.add_subplot(strips[1]), fig.add_subplot(outer[1])
    peak, cost, counts = {}, {}, {}
    for key, (variant, _, _) in FOOTWEAR.items():
        d = load(results, variant, 4.5)
        counts[key] = 0 if d is None else len(d["seed"])
        running = np.zeros(0, dtype=bool) if d is None else np.isin(d["gait"], ("Running", "Stiff Running"))
        peak[key] = np.array([]) if d is None else peak_vgrf(d)[running]
        cost[key] = np.array([]) if d is None or "umberger" not in d else d["umberger"][running]
    colors = {k: v[1] for k, v in FOOTWEAR.items()}
    labels = {k: v[2] for k, v in FOOTWEAR.items()}
    strip_plot(ax_vgrf, peak, colors, labels, "Peak vertical GRF [BW]")
    ref_mean, ref_std = reference_peak(4.5)
    ax_vgrf.axhline(ref_mean, color=REFERENCE, linestyle="--", linewidth=1.5, zorder=0)
    ax_vgrf.axhspan(ref_mean - ref_std, ref_mean + ref_std, color=REFERENCE, alpha=0.1, linewidth=0, zorder=0)
    strip_plot(ax_umberger, cost, colors, labels, "Metabolic cost [J/kg/m]")
    ax_render.axis("off")
    render = Path(results) / "figure4c_render.png"          # scripts/render_figure4c.py
    if render.exists():
        ax_render.imshow(plt.imread(render), aspect="equal", interpolation="lanczos")
    panel_label(fig, "c", ax_a, y_ax=ax_vgrf)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(PROJECT / "results"))
    parser.add_argument("--out", default=str(PROJECT / "results/figure4_biosym.png"))
    args = parser.parse_args()
    results = Path(args.results)

    data = {"metabolic": {s: load(results, "walking", s) for s in swept_speeds(results, "walking")},
            "effort": {s: load(results, "effort", s) for s in swept_speeds(results, "effort")}}
    with plt.rc_context({"font.size": FONT}):
        fig = plt.figure(figsize=(16, 10))
        outer = GridSpec(2, 1, height_ratios=[1, 0.85], hspace=0.55)
        top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[0.28 + 0.32, 0.28], wspace=0.3)
        left = GridSpecFromSubplotSpec(1, 2, subplot_spec=top[0, 0], width_ratios=[0.32, 0.28], wspace=0.3)
        ax_a = panel_cost(fig, left[0, 0], data)
        panel_grf(fig, left[0, 1], data)
        weak_counts = panel_weakness(fig, top[0, 1], results, ax_a)
        foot_counts = panel_footwear(fig, outer[1], results, ax_a)
        handles = [plt.Line2D([0], [0], color=REFERENCE, lw=2, label="Reference")] + \
                  [plt.Line2D([0], [0], color=COST_COLORS[c], lw=2, label=COST_SHORT_LABEL[c]) for c in COST_COLORS]
        fig.legend(handles=handles, fontsize=FONT, frameon=False, ncol=3, loc="center", bbox_to_anchor=(0.37, 0.46),
                   handlelength=1.2)
        for a in fig.axes:
            if a.axison:
                a.spines[["top", "right"]].set_visible(False)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("trials per set  metabolic:", {s: (0 if d is None else len(d["seed"])) for s, d in data["metabolic"].items()})
    print("                effort:   ", {s: (0 if d is None else len(d["seed"])) for s, d in data["effort"].items()})
    print("                weakness: ", weak_counts)
    print("                contact:  ", foot_counts)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
