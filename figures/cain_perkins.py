"""Steer torque per steer angle, Cain and Perkins (2012) (Sect. 15.2, Fig. cain)."""

import csv

import numpy as np
from figstyle import (
    BLACK,
    CM,
    COL_W,
    DATA,
    GREY,
    LIGHT,
    MK,
    ORANGE,
    panel,
    plt,
    read_json,
    save_paper,
)

C = read_json(DATA / "experiments" / "cain_perkins.json")
YLIM = (-16, 10)


def digitised(s):
    with open(DATA / "experiments" / "cain_perkins_2012_fig11.csv") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    xy = [
        (float(r["v2_m2_per_s2"]), float(r["T_over_delta_Nm_per_rad"]))
        for r in rows
        if r["subject"] == str(s)
    ]
    return np.array(xy).T


def subject(ax, s, data):
    v2 = np.linspace(0, 95, 200)
    nb = C["band"][str(s)]
    ax.fill_between(nb["v2"], nb["lo"], nb["hi"], color="#E4E4E4", lw=0, zorder=1)
    lin = C["linear"][f"{s}|Rigid+Jason"]
    ax.plot(v2, lin["m"] * v2 + lin["b"], color=BLACK, lw=1.0, zorder=3)
    m, b = C["table3"][str(s)]["normal"]
    ax.plot(v2, m * v2 + b, color=ORANGE, lw=1.0, zorder=3)
    vc2 = lin["v_c"] ** 2
    ax.axvline(vc2, color=GREY, lw=0.5, ls=(0, (1, 1.5)), zorder=0)
    ax.text(vc2 + 1.0, YLIM[1] - 0.6, r"$v_c^2$", fontsize=8, color=GREY, va="top")
    ax.axhline(0, color=LIGHT, lw=0.5, zorder=0)
    x, y = digitised(s)
    inside = y >= YLIM[0]
    ax.plot(x[inside], y[inside], "o", ms=3.2, mec=BLACK, **MK, zorder=4)
    for xi in x[~inside]:
        ax.plot(xi, YLIM[0] + 0.5, "v", ms=3.6, mec=BLACK, **MK, zorder=4, clip_on=False)
    ax.text(0.04, 0.95, f"Subject {s}", transform=ax.transAxes, fontsize=8, va="top")
    ax.set(xlim=(0, 95), ylim=YLIM, xticks=[0, 20, 40, 60, 80], xlabel=r"$V^2$ (m$^2$/s$^2$)")
    data[f"subject{s}"] = dict(
        digitised_v2=x,
        digitised_ratio=y,
        off_scale=x[~inside],
        linear_m=lin["m"],
        linear_b=lin["b"],
        v_c=lin["v_c"],
        fit_normal=[m, b],
        band=nb,
    )


def main():
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(COL_W, 5.6 * CM), sharey=True, gridspec_kw=dict(wspace=0.08)
    )
    data = {}
    subject(ax, 1, data)
    subject(bx, 2, data)
    ax.set_ylabel(r"$T/\delta$ (N m/rad)")
    ax.text(94, -1.5, "Linear model", fontsize=8, ha="right", va="center")
    ax.text(60, 4.3, "Nonlinear model", fontsize=8, ha="right", va="center", color=GREY)
    ax.text(94, -12.0, "Measured fit", fontsize=8, ha="right", va="center", color=ORANGE)
    bx.text(93, 6.0, "Measured fit", fontsize=8, ha="right", color=ORANGE)
    bx.text(89, -0.75, "Linear model", fontsize=8, ha="right", va="center")
    panel(ax, "a", x=-0.2)
    panel(bx, "b", x=-0.05)
    save_paper(fig, "cain_perkins", data)
    for k, d in data.items():
        print(k, f"m {d['linear_m']:.4f} b {d['linear_b']:.3f} v_c {d['v_c']:.2f}")


if __name__ == "__main__":
    main()
