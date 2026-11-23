"""Largest stable lean and lateral acceleration of the 23 measured sets (fig:measured-overview,
supplement Sect. "Results for each set").
"""

import numpy as np
from figstyle import BLACK, CM, FULL_W, GREY, MK, YSTAR, all_results, nice_name, plt, save_paper

STY = {
    "free": dict(marker="o", ms=3.8, mec=BLACK, ls="none", **MK),
    "constant": dict(marker="D", ms=3.0, mec=GREY, ls="none", **MK),
}


def rows():
    out = []
    for name, d in all_results().items():
        if d.get("group") != "rider":
            continue
        f, c = d["free"], d["constant"]
        out.append(
            (nice_name(name), f["max_lean_deg"], c["max_lean_deg"], f["max_ay_g"], c["max_ay_g"])
        )
    return sorted(out, key=lambda r: r[1])


def column(ax, y, a, b, xlabel, xlim, xticks):
    for yi, ai, bi in zip(y, a, b):
        ax.plot([ai, bi], [yi, yi], color="#D0D0D0", lw=0.8, zorder=1)
    ax.plot(b, y, **STY["constant"], zorder=2)
    ax.plot(a, y, **STY["free"], zorder=3)
    ax.set(xlim=xlim, xticks=xticks, xlabel=xlabel, ylim=(-0.8, len(y) - 0.2))
    ax.grid(axis="x", color="#E8E8E8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    r = rows()
    y = np.arange(len(r))
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(FULL_W, 8.0 * CM), sharey=True, gridspec_kw=dict(wspace=0.08)
    )
    column(ax, y, [q[1] for q in r], [q[2] for q in r], YSTAR, (5, 36), [10, 20, 30])
    column(
        bx,
        y,
        [q[3] for q in r],
        [q[4] for q in r],
        r"Largest stable $a_y/g$",
        (0.1, 0.76),
        [0.2, 0.4, 0.6],
    )
    ax.set_yticks(y, [q[0] for q in r])
    ax.tick_params(axis="y", length=0, labelsize=7.5)
    bx.tick_params(axis="y", length=0)
    for i, (k, text) in enumerate([("free", "Free rolling"), ("constant", "Constant speed")]):
        bx.plot([0.55], [0.16 - i * 0.075], transform=bx.transAxes, clip_on=False, **STY[k])
        bx.text(0.60, 0.16 - i * 0.075, text, transform=bx.transAxes, fontsize=8, va="center")
    keys = ("lean_free", "lean_constant", "ay_free", "ay_constant")
    save_paper(fig, "measured_overview", {q[0]: dict(zip(keys, q[1:])) for q in r}, width=None)
    for q in r:
        print(f"{q[0]:40s} {q[1]:6.2f} {q[2]:6.2f} {q[3]:.3f} {q[4]:.3f}")


if __name__ == "__main__":
    main()
