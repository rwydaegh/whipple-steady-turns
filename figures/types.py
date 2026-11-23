"""fig:types, torque coefficient, trail and maps (Sect. 10, "Closed and open regions")."""

import json

import numpy as np
from figstyle import (
    BLACK,
    CAPSIZE,
    CM,
    COL_W,
    DATA,
    GREY,
    MK,
    ORANGE,
    STABLE,
    STABLE_ALPHA,
    STABLE_TEXT,
    WEAVE,
    XV,
    YLEAN,
    YSTAR,
    panel,
    plt,
    read_json,
    save_paper,
)
from scipy import ndimage

TYPES = DATA / "types"
COLOR = {"benchmark": BLACK, "A": ORANGE}
MAPS = [(0.080, "Closed"), (0.190, "Near opening"), (0.220, "Open")]


def draw_tau3(ax, curves):
    ax.axhline(0, color="k", lw=0.5)
    for d, col in zip(curves, (BLACK, GREY)):
        ax.plot(d["V"], d["tau3"], color=col, lw=1.0)
        ax.plot(d["V3"], 0, "o", ms=3.5, mec=col, zorder=5, **MK)
        ax.plot([d["v_c"]], [0], "^", ms=4.5, mfc=CAPSIZE, mec=CAPSIZE, clip_on=False, zorder=6)
    b, t = curves
    ax.text(8.4, 2.6, "Trail 80 mm", fontsize=8, ha="left")
    ax.text(8.4, 1.9, "and 200 mm", fontsize=8, ha="left", color=GREY)
    ax.annotate(
        r"$V_3$",
        xy=(b["V3"], 0),
        xytext=(b["V3"] - 0.9, 0.9),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=0, shrinkB=2),
    )
    ax.annotate(
        r"$v_c$, 80 mm",
        xy=(b["v_c"], 0),
        xytext=(b["v_c"] - 0.3, -1.3),
        fontsize=8,
        ha="right",
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=0, shrinkB=3),
    )
    ax.annotate(
        r"$v_c$, 200 mm",
        xy=(t["v_c"], 0),
        xytext=(t["v_c"] + 0.2, 0.9),
        fontsize=8,
        color=GREY,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY, shrinkA=0, shrinkB=3),
    )
    ax.set(
        xlim=(3, 12), ylim=(-2, 5), xticks=[4, 6, 8, 10, 12], xlabel=XV, ylabel=r"$\tau_3$ (N m)"
    )


def trail_series(tops, crit, name):
    c = 1000 * np.array([r["c"] for r in tops[name]])
    top = np.array([r["lean_deg"] for r in tops[name]])
    o = crit[f"{name} trail"]["free"]
    return c, top, 1000 * o["opening"], o["lean_deg"]


def draw_trail(ax, tops, crit):
    for name, label in (("benchmark", "Benchmark"), ("A", "Bicycle A")):
        c, top, cs, ls = trail_series(tops, crit, name)
        col = COLOR[name]
        ax.plot(np.r_[c, cs], np.r_[top, ls], color=col, lw=1.0, zorder=3)
        ax.plot(c, top, "o", ms=3.0, mec=col, zorder=4, **MK)
        ax.plot(cs, ls, "o", ms=4.0, mfc=col, mec=col, zorder=5)
        ax.annotate(
            "",
            xy=(cs, 44),
            xytext=(cs, ls + 1.2),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, mutation_scale=7),
        )
        ax.text(cs, 45, "Opens", color=col, ha="center", va="bottom", fontsize=8)
        if name == "A":
            ax.text(250, 16.5, label, color=col, fontsize=8, ha="left", va="top")
        else:
            ax.text(176, 30, label, color=col, fontsize=8, ha="right", va="center")
    ax.set(
        xlim=(50, 400),
        ylim=(0, 50),
        xticks=[100, 200, 300, 400],
        yticks=[0, 10, 20, 30, 40],
        xlabel=r"Trail $c$ (mm)",
        ylabel=YSTAR,
    )


def draw_map(ax, c, near=3):
    d = np.load(TYPES / f"benchmark_c{c:.3f}.npz")
    Vs, phis, comp = d["Vs"], d["phis"], d["stable"]
    X, Y = np.meshgrid(Vs, phis, indexing="ij")
    Xf, Yf = np.c_[X[:, :1], X], np.c_[np.zeros((len(Vs), 1)), Y]
    Cf = np.c_[comp[:, :1], comp].astype(float)
    ax.contourf(Xf, Yf, Cf, levels=[0.5, 1.5], colors=[STABLE], alpha=STABLE_ALPHA, zorder=1)
    edge = ndimage.binary_dilation(comp, iterations=near) & ~ndimage.binary_erosion(
        comp, iterations=near
    )
    a4 = d["a4"].astype(float)
    ax.contour(
        X, Y, np.where(edge, a4, np.nan), levels=[0], colors=[CAPSIZE], linewidths=1.0, zorder=3
    )
    D3 = np.where(edge & (a4 > 0), d["D3"].astype(float), np.nan)
    ax.contour(X, Y, D3, levels=[0], colors=[WEAVE], linewidths=1.0, zorder=3)
    ax.set(xlim=(2, 14), ylim=(0, 40), xticks=[4, 8, 12], yticks=[0, 10, 20, 30, 40], xlabel=XV)
    meta = json.loads(str(d["meta"]))
    return dict(
        c=c,
        grid_top_deg=float(phis[np.flatnonzero(comp.any(axis=0)).max()]),
        v_w=meta["v_w"],
        v_c=meta["v_c"],
    )


def main():
    s = read_json(TYPES / "trail_series.json")
    crit = read_json(TYPES / "critical_trails.json")
    fig = plt.figure(figsize=(COL_W, 10.4 * CM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.95], hspace=0.55)
    top = gs[0].subgridspec(1, 2, wspace=0.38)
    bot = gs[1].subgridspec(1, 3, wspace=0.1)
    ax = fig.add_subplot(top[0])
    draw_tau3(ax, s["tau3"])
    panel(ax, "a", x=-0.27)
    bx = fig.add_subplot(top[1])
    draw_trail(bx, s["tops"], crit)
    panel(bx, "b", x=-0.27)
    cx = [fig.add_subplot(bot[k]) for k in range(3)]
    maps = []
    for k, (a, (c, kind)) in enumerate(zip(cx, MAPS)):
        maps.append(draw_map(a, c))
        a.text(0.04, 0.97, kind, transform=a.transAxes, fontsize=8, va="top")
        a.text(
            0.04,
            0.85,
            rf"$c={1000 * c:.0f}$ mm",
            transform=a.transAxes,
            fontsize=8,
            va="top",
            color=GREY,
        )
        if k > 0:
            a.set_yticklabels([])
        else:
            a.set_ylabel(YLEAN)
    cx[1].text(7.0, 8.0, "Stable", color=STABLE_TEXT, fontsize=8, ha="center")
    cx[1].text(9.3, 15.0, "Capsize", color=CAPSIZE, fontsize=8)
    cx[1].text(2.1, 4.0, "Weave", color=WEAVE, fontsize=8)
    panel(cx[0], "c", x=-0.3)
    data = dict(
        tau3=[{k: d[k] for k in ("c", "V3", "v_c", "tau3_at_vc")} for d in s["tau3"]],
        opening_mm={n: 1000 * crit[f"{n} trail"]["free"]["opening"] for n in ("benchmark", "A")},
        maps=maps,
        plant="free rolling",
    )
    save_paper(fig, "types", data)
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
