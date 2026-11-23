"""Region of stable turns of three measured bicycles with and without a rider (fig:rider-effect,
supplement Sect. "Effect of the rider").
"""

import numpy as np
from figstyle import (
    BARE,
    CAPSIZE,
    CM,
    FULL_W,
    STABLE_TEXT,
    WEAVE,
    XV,
    YLEAN,
    boundary,
    fill_region,
    load_result,
    plt,
    save_paper,
    tip_marker,
)

PAIRS = [("Crescendo", "Crescendo+Jason"), ("Stratos", "Stratos+Jason"), ("Rigid", "Rigid+Jason")]


def draw(ax, bare, rider):
    Vh, lh, Vf, lf = boundary(load_result(bare))
    fill_region(ax, Vh, lh, Vf, lf, color=BARE, alpha=0.28)
    ax.plot(Vh, lh, color=BARE, lw=1.0, zorder=2)
    ax.plot(Vf, lf, color=BARE, lw=1.0, zorder=2)
    tip_marker(ax, Vh[-1], lh[-1], color=BARE)
    Rh, rh, Rf, rf = boundary(load_result(rider))
    fill_region(ax, Rh, rh, Rf, rf)
    ax.plot(Rh, rh, color=WEAVE, zorder=3)
    ax.plot(Rf, rf, color=CAPSIZE, zorder=3)
    tip_marker(ax, Rh[-1], rh[-1])
    return {
        name: dict(
            weave_V=a, weave_lean=b, capsize_V=c, capsize_lean=d, largest_lean=float(np.max(d))
        )
        for name, (a, b, c, d) in ((bare, (Vh, lh, Vf, lf)), (rider, (Rh, rh, Rf, rf)))
    }


def main():
    fig, axs = plt.subplots(
        1, 3, figsize=(FULL_W, 4.8 * CM), sharey=True, gridspec_kw=dict(wspace=0.1)
    )
    data = {}
    for k, (ax, (bare, rider)) in enumerate(zip(axs, PAIRS)):
        data.update(draw(ax, bare, rider))
        ax.set(xlim=(2, 10.5), ylim=(0, 21), xticks=[4, 6, 8, 10])
        ax.text(0.97, 0.97, bare, transform=ax.transAxes, ha="right", va="top", fontsize=8)
        if k == 0:
            ax.set_ylabel(YLEAN)
            ax.text(7.4, 5.0, "With rider", color=STABLE_TEXT, fontsize=8, ha="center")
            ax.text(
                2.8,
                8.5,
                "Bicycle\nalone",
                color="#505050",
                fontsize=8,
                ha="center",
                va="top",
                linespacing=1.1,
            )
    fig.supxlabel(XV, fontsize=9, y=-0.06)
    save_paper(fig, "rider_effect", data, width=None)
    print({k: round(v["largest_lean"], 2) for k, v in data.items()})


if __name__ == "__main__":
    main()
