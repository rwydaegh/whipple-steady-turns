"""Steer torque that holds a steady turn of the benchmark bicycle (Fig. 1) and the peak of the
hands-free lobe for the three bicycles (Sect. 3)."""

import numpy as np

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.bicycles import BICYCLES
from whipple.model import Whipple
from whipple.output import options, save_json


def lobe_peak(bike):
    curve = st.hands_free(bike)
    x = max(curve, key=lambda x: bike.turn(*x).ay)
    x = st.boundary_extremum(bike, lambda x: bike.turn_spin_torque(*x)[1], lambda t: t.ay, x)
    peak = bike.turn(*x)
    return {"peak a_y/g": peak.ay / ps.G, "at V (m/s)": peak.V, "lean (deg)": np.degrees(peak.phi)}


def main():
    args = options(__doc__)
    bike = Whipple(BICYCLES["benchmark"])
    save_json("torque_map", {name: lobe_peak(Whipple(p)) for name, p in BICYCLES.items()})

    n = 70 if args.quick else 180
    phis = np.linspace(5e-4, 0.75, n)
    deltas = np.geomspace(1e-4, 1.3, int(1.5 * n))
    V, ay, T = st.turn_grid(bike, phis, deltas)
    t = st.tongue(bike, ds_max=1e-2 if args.quick else 4e-3)

    fig, ax = ps.figure(ps.HALF, 5.6 * ps.CM)
    hopf = np.array([[x.V, x.ay] for x in (bike.turn(*p) for p in t.hopf)]) / [1, ps.G]
    fold = np.array([[x.V, x.ay] for x in (bike.turn(*p) for p in t.fold)]) / [1, ps.G]
    ps.region(ax, *hopf.T, *fold.T)
    cs = ax.contour(V, ay / ps.G, T, levels=[-1.0, -0.5, -0.25, 0.25, 0.5], colors=ps.CONTOUR,
                    linewidths=0.5, zorder=1)  # fmt: skip
    ps.contour_labels(ax, cs, 0.74, x_min=1.5, x_max=9.8)
    # Near upright T is zero to rounding, so the hands-free turns are drawn from the grid only
    # above 0.02 g.
    T0 = np.where(ay < 0.02 * ps.G, np.nan, T)
    ax.contour(V, ay / ps.G, T0, levels=[0.0], colors=ps.HANDS_FREE, linewidths=ps.LW, zorder=3)
    lobe = np.array([[x.V, x.ay / ps.G] for x in (bike.turn(*p) for p in st.hands_free(bike, 0.1))])
    ax.plot(*lobe.T, color=ps.HANDS_FREE, zorder=3)
    ax.set(xlim=(1, 10), ylim=(0, 0.8), xticks=[2, 4, 6, 8, 10], xlabel=ps.XV, ylabel=ps.YAY)
    ps.label(ax, 4.9, 0.08, "Stable", color=ps.STABLE_TEXT, ha="center")
    ps.label(ax, 6.15, 0.05, r"$T=0$")
    ps.label(ax, 8.6, 0.12, r"$T>0$", ha="center")
    ps.label(ax, 2.6, 0.08, r"$T<0$", ha="center")
    ps.save(fig, "torque_map")


if __name__ == "__main__":
    main()
