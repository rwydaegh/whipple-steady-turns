"""Recovery and capsize after a kick (Sect. 15.1, "Time simulations", Fig. sim)."""

import numpy as np
from figstyle import (
    CAPSIZE,
    CM,
    COL_W,
    DATA,
    GREY,
    STABLE_TEXT,
    YLEAN,
    panel,
    plt,
    read_json,
    save_paper,
)

K = read_json(DATA / "simulations" / "kicks.json")
RUNS = [("stable_turn", STABLE_TEXT), ("capsize", CAPSIZE)]
KICK = 2.0


def main():
    data = {n: {k: np.asarray(v) for k, v in K[n].items() if isinstance(v, list)} for n, _ in RUNS}
    fig = plt.figure(figsize=(COL_W, 5.0 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.8], wspace=0.42)
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))
    for n, col in RUNS:
        d = data[n]
        ax.plot(d["t"], d["lean_deg"], color=col, lw=1.0)
        bx.plot(d["t"], d["speed"], color=col, lw=1.0)
        tr = d["track_rear_xy"]  # model axes: x forward, y to the right
        cx.plot(tr[:, 1], tr[:, 0], color=col, lw=1.0)
        k = np.searchsorted(d["t"], KICK)
        cx.plot(tr[k, 1], tr[k, 0], "|", color="k", ms=4, mew=0.8, zorder=5)
    for a in (ax, bx):
        a.axvline(KICK, color=GREY, lw=0.5, ls=":", zorder=0)
        a.set(xlim=(0, 18), xticks=[0, 5, 10, 15], xlabel=r"Time $t$ (s)")
    ax.text(KICK + 0.3, 44, "Kick", fontsize=8, color=GREY, va="top")
    ax.set(ylim=(0, 46), yticks=[0, 15, 30, 45], ylabel=YLEAN)
    c, s = data["capsize"], data["stable_turn"]
    ax.text(c["t"][-1] + 0.4, 43, "Capsize", color=CAPSIZE, fontsize=8, va="top")
    ax.text(
        17.8,
        s["lean_deg"][-1] + 2.5,
        "Recovery",
        color=STABLE_TEXT,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    bx.set(ylim=(4.4, 6.2), yticks=[4.5, 5.0, 5.5, 6.0], ylabel=r"Speed $V$ (m/s)")
    bx.text(
        c["t"][-1] - 0.3,
        c["speed"][-1],
        "Capsize",
        color=CAPSIZE,
        fontsize=8,
        ha="right",
        va="center",
    )
    bx.text(
        17.8,
        s["speed"][-1] + 0.06,
        "Recovery",
        color=STABLE_TEXT,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    cx.plot(0, 0, "o", ms=3, color="k", zorder=5)
    cx.text(-2.3, 0, "Start", fontsize=8, ha="right", va="center")
    cx.text(8.75, 0.6, "Capsize", fontsize=7.5, color=CAPSIZE, ha="center", va="center")
    cx.text(8.4, -10.2, "Recovery", fontsize=8, color=STABLE_TEXT, ha="center", va="top")
    cx.set_aspect("equal")
    cx.axis("off")
    cx.plot([15, 20], [-11.0, -11.0], color="k", lw=1.0)
    cx.text(17.5, -11.6, "5 m", ha="center", va="top", fontsize=8)
    cx.set(xlim=(-5, 20), ylim=(-13, 14))
    panel(ax, "a", x=-0.3)
    panel(bx, "b", x=-0.3)
    fig.canvas.draw()
    fig.text(cx.get_position().x0 - 0.01, ax.get_position().y1, "(c)", fontsize=9, va="bottom")
    save_paper(fig, "simulations", data)
    for n, d in data.items():
        print(
            n,
            "end t",
            d["t"][-1],
            "lean",
            round(d["lean_deg"][-1], 3),
            "speed",
            round(d["speed"][-1], 3),
        )


if __name__ == "__main__":
    main()
