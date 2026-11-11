"""
Figure 3 of the BiomechPriorVAE paper with the biosym solutions as "Ours": 2-Wasserstein distance to the reference per
model, walking versus running, joint angles versus ground reaction forces.

    python scripts/plot_figure3.py [--w2 data/w2/per_speed.csv] [--out results/figure3_biosym.png]

Same layout as ``plot/figure03.py`` of the BiomechPriorVAE repository: two stacked panels (angles W2 [deg] on top, GRF W2
[BW] below), one slot per model (Ours, PredSim, GGN, GaitDynamics, GaitEncoder), walking (0.8-1.6 m/s, circles) left and
running (2.5-4.5 m/s, triangles) right in each slot, one jittered point per swept speed and the mean +/- SD of the band.
A model without an estimable speed in a band leaves a gap. GaitEncoder has no ground reaction forces, so its slot of the
lower panel hosts the legend. The distances are written by ``compute_w2.py``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

PROJECT = Path(__file__).resolve().parents[1]
WALK_SPEEDS = (0.8, 1.0, 1.2, 1.4, 1.6)
RUN_SPEEDS = (2.5, 3.5, 4.5)


def _lighten(hex_color, amount):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(rgb + (1 - rgb) * amount)


COLORS = {"Ours": "#c9b437", "PS": "#C56C14", "GGN": "#C85293", "GD": "#009E73", "GE": _lighten("#C20707", 0.45)}
NAMES = {"Ours": "Ours", "PS": "PS", "GGN": "GGN", "GD": "GD", "GE": "GE"}
PHYSICS = {"Ours", "PS", "GGN"}
MARKER = {"walk": "o", "run": "^"}
DX = {"walk": -0.16, "run": 0.16}
JITTER = 0.06
DRAWN_WIDTH_IN = 8
PRINTED_WIDTH_IN = 7.087 / 2
FONT = 6 * DRAWN_WIDTH_IN / PRINTED_WIDTH_IN


def band_values(table, model, speeds, key):
    """Finite per-speed distances of a model within a speed band."""
    rows = table[(table["model"] == model) & table["speed"].isin(speeds)]
    values = rows[key].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def draw_panel(ax, table, models, key, hide_labels=()):
    rng = np.random.default_rng(0)
    for i, model in enumerate(models):
        for band, speeds in (("walk", WALK_SPEEDS), ("run", RUN_SPEEDS)):
            values = band_values(table, model, speeds, key)
            if len(values) == 0:
                continue
            x0 = i + DX[band]
            if len(values) == 1:
                ax.scatter([x0], values, color=COLORS[model], alpha=0.55, s=14, linewidth=0, marker=MARKER[band])
                continue
            ax.scatter(x0 + rng.uniform(-JITTER, JITTER, size=len(values)), values, color=COLORS[model], alpha=0.55,
                       s=14, linewidth=0, marker=MARKER[band])
            ax.errorbar([x0], [values.mean()], yerr=[values.std()], color=COLORS[model], marker=MARKER[band],
                        markersize=6, markeredgecolor="k", markeredgewidth=0.6, capsize=3, linestyle="none", zorder=5)
    shown = [i for i, m in enumerate(models) if m not in hide_labels]
    ax.set_xticks(shown)
    ax.set_xticklabels([NAMES[models[i]] for i in shown])
    ax.set_xlim(-0.6, len(models) - 0.4)


def print_summary(table, models):
    for key, label in (("ang_w2", "Angles W2 [deg]"), ("grf_w2", "GRF W2 [BW]")):
        print(f"\n{label}\n model   band   mean     SD   n")
        for m in models:
            for band, speeds in (("walk", WALK_SPEEDS), ("run", RUN_SPEEDS)):
                v = band_values(table, m, speeds, key)
                print(f"{m:>6} {band:>6} " + (f"{v.mean():6.3f} {v.std():6.3f} {len(v):3d}" if len(v) else "    --     --   0"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w2", default=str(PROJECT / "data/w2/per_speed.csv"))
    parser.add_argument("--out", default=str(PROJECT / "results/figure3_biosym.png"))
    args = parser.parse_args()
    table = pd.read_csv(args.w2)
    models = list(COLORS)
    print_summary(table, models)

    with plt.rc_context({"font.size": FONT}):
        fig, (ax_ang, ax_grf) = plt.subplots(2, 1, figsize=(DRAWN_WIDTH_IN, DRAWN_WIDTH_IN * 0.85))
        draw_panel(ax_ang, table, models, "ang_w2")
        ax_ang.set_ylabel("Angles W2 [deg]")
        ax_ang.yaxis.set_major_locator(MaxNLocator(integer=True))
        draw_panel(ax_grf, table, models, "grf_w2", hide_labels={"GE"})
        ax_grf.set_ylabel("GRF W2 [BW]")
        ax_grf.yaxis.set_major_locator(MultipleLocator(0.1))
        ax_grf.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        for ax in (ax_ang, ax_grf):
            ax.spines[["top", "right", "bottom"]].set_visible(False)

        handles = [plt.Line2D([0], [0], marker=MARKER[b], color="0.3", linestyle="none", markersize=6, label=l)
                   for b, l in (("walk", "Walking"), ("run", "Running"))]
        lo, hi = ax_grf.get_ylim()
        ax_grf.legend(handles=handles, loc="center", bbox_to_anchor=(models.index("GE"), (lo + hi) / 2),
                      bbox_transform=ax_grf.transData, frameon=False, fontsize=FONT * 0.85)
        fig.tight_layout()

        physics = [i for i, m in enumerate(models) if m in PHYSICS]
        p_lo, p_hi = min(physics) - 0.5, max(physics) + 0.5
        top, bottom = ax_ang.get_position(), ax_grf.get_position()
        blended = mtransforms.blended_transform_factory(ax_ang.transData, fig.transFigure)
        for ax in (ax_ang, ax_grf):
            ax.patch.set_visible(False)
        fig.add_artist(Rectangle((p_lo, bottom.y0), p_hi - p_lo, top.y1 - bottom.y0, transform=blended,
                                 facecolor="0.94", edgecolor="none", zorder=-10))
        ax_ang.text((p_lo + p_hi) / 2, top.y1 - 0.015, "physics-based", transform=blended, ha="center", va="top",
                    fontsize=FONT * 0.8)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
