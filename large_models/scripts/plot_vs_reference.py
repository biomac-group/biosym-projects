"""
The metabolic walking solutions at 1.3 m/s of the three models against the 1.2 and 1.4 m/s reference.

    python scripts/plot_vs_reference.py [--out results/figure_metabolic_vs_reference.png]

Layout and data of ``3d_simulations/scripts/plot_figure1.py`` panel a: hip / knee / ankle angle [deg, gait3d sign: knee
flexion negative] in the left column, Fx (forward) and Fy (vertical) ground reaction force of the right foot [body weights]
in the right column, over one gait cycle. The reference (``data/reference/gait_speed_mean_var.npz``) is drawn as mean with
+-1 SD band at 1.2 m/s (solid) and 1.4 m/s (dashed). The three runs are RunMad-SIPP (``3d_simulations``, best converged seed
of ``walking/v1.30``), MS-Human-700 and MyoFullBody (``results/MODEL/walking_metabolic_v1.30_seed1.pkl``); each is rolled to
its right heel strike and then shifted once (steps of a quarter frame) to the best waveform correlation of its ground
reaction forces with the mean of the two reference curves (``plot_figure1.best_shift``). Cadence and duty factor are in the
legend.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gait_vs_reference as gvr  # noqa: E402  (also puts the 3d_simulations plot libraries on the path)
import data_utils as du  # noqa: E402
import plot_figure1 as pf  # noqa: E402

from biosym.model.model import BiosymModel  # noqa: E402

COLORS = {"RunMad-SIPP": "#c9b437", "MS-Human-700": "#0072B2", "MyoFullBody": "#7B3294", "reference": "#000000"}
RUNS = {"RunMad-SIPP": "SIPP metabolic", "MS-Human-700": "MS-Human metabolic", "MyoFullBody": "MyoFullBody metabolic"}
LABELS = ["Hip flexion [deg]", "Knee angle [deg]", "Ankle angle [deg]", "Fx [BW]", "Fy [BW]"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(gvr.LARGE / "results" / "figure_metabolic_vs_reference.png"))
    args = parser.parse_args()

    refs, sds = {}, {}
    for speed in gvr.REFERENCE_SPEEDS:
        r = du.get_reference_data(speed, root_path=str(gvr.PROJECTS / "3d_simulations/data/reference"))
        refs[speed] = np.array([r[k]["mean"] for k in ("hip", "knee", "ankle", "Fx", "Fy")]).T
        sds[speed] = np.array([r[k]["std"] for k in ("hip", "knee", "ankle", "Fx", "Fy")]).T
    target = np.mean([refs[s] for s in gvr.REFERENCE_SPEEDS], axis=0)       # the 1.3 m/s reference: mean of 1.2 and 1.4

    fonts = {"font.size": pf.FONT} if hasattr(pf, "FONT") else {}
    plt.rcParams.update(fonts)
    fig = plt.figure(figsize=(9.5, 7.2))
    grid = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)] + [fig.add_subplot(grid[i, 1]) for i in range(2)]
    x = np.arange(100)
    for speed, ls in zip(gvr.REFERENCE_SPEEDS, ("-", "--")):
        for c, ax in enumerate(axes):
            ax.plot(refs[speed][:, c], color=COLORS["reference"], ls=ls, lw=2, zorder=1)
            ax.fill_between(x, refs[speed][:, c] - sds[speed][:, c], refs[speed][:, c] + sds[speed][:, c],
                            color=COLORS["reference"], alpha=0.08, lw=0, zorder=1)

    models, handles = {}, []
    for label, run in RUNS.items():
        _, yaml, result, knee_sign = next(r for r in gvr.RUNS if r[0] == run)
        if yaml not in models:
            models[yaml] = BiosymModel(str(yaml), compile_eom=False, fit_muscle_paths=False)
        sim, step = gvr.cycle(models[yaml], result, knee_sign)
        duty = pf.duty_factor(sim[:, 4], step)                    # on the cycle rolled to heel strike, before the shift
        print(f"{label}: {Path(result).name} of {Path(result).parent.name}, step {step:.3f} s, duty {duty:.3f}", flush=True)
        shift = pf.best_shift(sim, target, channels=pf.ALIGN_CHANNELS)
        sim = pf.roll_frac(sim, shift, axis=0)
        for c, ax in enumerate(axes):
            ax.plot(sim[:, c], color=COLORS[label], lw=2, zorder=3)
        handles.append(Line2D([0], [0], color=COLORS[label], lw=2,
                              label=f"{label}  ({60.0 / step:.0f} steps/min, duty {duty:.2f})"))

    for ax, label in zip(axes, LABELS):
        ax.set_ylabel(label)
        ax.set_xlim(0, 100)
    axes[1].set_yticks([-90, -45, 0])
    axes[3].set_yticks([-0.4, -0.2, 0, 0.2])
    axes[3].set_ylim(-0.5, 0.3)
    for ax in (axes[2], axes[4]):
        ax.set_xlabel("Gait cycle [%]")
    for ax in axes[:2] + axes[3:4]:
        ax.tick_params(labelbottom=False)
    legend = fig.add_subplot(grid[2, 1])
    legend.axis("off")
    ref_handles = [Line2D([0], [0], color="k", lw=2, ls=ls, label=f"Reference {s} m/s (mean ± SD)")
                   for s, ls in zip(gvr.REFERENCE_SPEEDS, ("-", "--"))]
    legend.legend(handles=ref_handles + handles, loc=(-0.25, -0.05), fontsize=pf.FONT if hasattr(pf, "FONT") else 8,
                  frameon=False, handlelength=1.4)
    fig.suptitle("Metabolic walking at 1.3 m/s vs reference (BiomechPriorVAE + metabolic cost)", y=0.97)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("written:", out, "and", out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
