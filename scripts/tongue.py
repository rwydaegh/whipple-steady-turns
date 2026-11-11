"""Region of stable turns at constant steer torque (Sect. 4.2): Figs. 3 and 7, Tables 1 and D2."""

import numpy as np

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.bicycles import BICYCLES
from whipple.model import Whipple
from whipple.output import options, save_json


def summary(bike, t):
    bt = bike.turn(*t.bt)
    lean = st.largest_stable(bike, t, lambda x: x.phi)
    ay = st.largest_stable(bike, t, lambda x: x.ay)
    others = sorted(st.eigenvalues(bike, *t.bt), key=abs)[2:]
    return {
        "upright weave speed (m/s)": t.v_weave,
        "upright capsize speed (m/s)": t.v_capsize,
        "BT lean (deg)": np.degrees(bt.phi),
        "BT steer (rad)": bt.delta,
        "BT speed V (m/s)": bt.V,
        "BT radius R (m)": bt.R,
        "BT steer torque (N m)": bt.T,
        "BT lateral acceleration a_y/g": bt.ay / ps.G,
        "BT other eigenvalues (1/s)": [z.real for z in others],
        "BT friction demand": bike.friction_demand(bt),
        "largest stable lean (deg)": np.degrees(lean.phi),
        "largest stable lean at V (m/s)": lean.V,
        "largest stable a_y/g": ay.ay / ps.G,
        "largest stable a_y at V (m/s)": ay.V,
    }


def benchmark_figure(bike, t, quick):
    n = 50 if quick else 120
    phis = np.linspace(5e-4, 0.36, n)
    deltas = np.geomspace(1e-3, 1.0, int(1.4 * n))
    V, ay, T = st.turn_grid(bike, phis, deltas)
    lean = np.degrees(phis)[:, None] * np.ones_like(V)
    # Hide the steep torque field left of the hands-free curve and in the top-left corner.
    T = np.where((T > 0) | ((V < 2.35) & (lean > 17)), np.nan, T)
    free = st.hands_free(bike, phi_max=0.4)
    fig, axs = ps.figure(ps.FULL, 6.2 * ps.CM, 1, 2)
    fig.subplots_adjust(wspace=0.32)
    for ax, kind, ymax in zip(axs, ("lean", "ay"), (20, 0.35)):
        Y = lean if kind == "lean" else ay / ps.G
        cs = ax.contour(V, Y, T, levels=[-0.8, -0.6, -0.4, -0.2], colors=ps.CONTOUR,
                        linewidths=0.5, zorder=0)  # fmt: skip
        ps.contour_labels(ax, cs, 0.9 * ymax, x_min=2.3, fmt="${:.1f}$")
        ax.plot(*ps.tongue_curve(bike, free, kind), color=ps.HANDS_FREE, zorder=2)
        x_bt, y_bt = ps.draw_tongue(ax, bike, t, kind)
        ax.set(
            xlim=(2, 7), ylim=(0, ymax), xlabel=ps.XV, ylabel=ps.YLEAN if kind == "lean" else ps.YAY
        )
        ps.label(ax, x_bt + 0.05, y_bt - 0.05 * ymax, "BT", ha="center", va="top")
        ps.label(ax, 3.55, 0.25 * ymax, "Hopf", color=ps.HOPF, ha="right")
        ps.label(ax, 6.12, 0.25 * ymax, "Fold", color=ps.FOLD)
        ps.label(ax, 5.0, 0.16 * ymax, "Stable", color=ps.STABLE_TEXT, ha="center")
        ps.label(ax, 6.2, 0.9 * ymax, r"$T=0$")
    axs[0].set_yticks([0, 5, 10, 15, 20])
    axs[1].set_yticks([0, 0.1, 0.2, 0.3])
    ps.panel(axs[0], "a")
    ps.panel(axs[1], "b")
    ps.save(fig, "tongue_benchmark")


def pair_figure(bikes, tongues):
    fig, ax = ps.figure(ps.HALF, 5.6 * ps.CM)
    for name in ("A", "B"):
        t = tongues[name]
        ps.draw_tongue(ax, bikes[name], t, color=ps.BIKE[name], fill=False, beyond=False)
        top = max(t.fold, key=lambda x: x[0])
        V = bikes[name].turn(*top).V
        ps.label(ax, V, np.degrees(top[0]) + (1.0 if name == "B" else -1.1), name,
                 color=ps.BIKE[name], ha="center")  # fmt: skip
    ax.set(xlim=(3.0, 6.4), ylim=(0, 18), xlabel=ps.XV, ylabel=ps.YLEAN, yticks=[0, 5, 10, 15])
    ps.label(ax, 4.62, 3.0, "Hopf", ha="right")
    ps.label(ax, 6.18, 3.0, "Fold")
    ps.save(fig, "tongue_pairs")


def main():
    args = options(__doc__)
    ds = 1e-2 if args.quick else 4e-3
    bikes = {name: Whipple(p) for name, p in BICYCLES.items()}
    tongues = {name: st.tongue(bike, ds_max=ds) for name, bike in bikes.items()}
    save_json("tongue", {name: summary(bikes[name], t) for name, t in tongues.items()})
    benchmark_figure(bikes["benchmark"], tongues["benchmark"], args.quick)
    pair_figure(bikes, tongues)


if __name__ == "__main__":
    main()
