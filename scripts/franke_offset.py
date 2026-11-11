"""Turns held by a lateral offset of the rear-frame mass centre at zero steer torque, the trim of
Franke et al. (1990), compared with turns at constant steer torque (Sect. 4.4, Fig. 4b)."""

import numpy as np

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.output import options, save_json


def offset_grid(bike, phis, deltas):
    """Speed V (m/s) and offset W (m) of the offset-trimmed turns on a (lean, steer) grid."""
    V, W = (np.full((len(phis), len(deltas)), np.nan) for _ in range(2))
    for i, phi in enumerate(phis):
        for j, delta in enumerate(deltas):
            try:
                w = st.offset_trim(bike, phi, delta)
            except RuntimeError:
                continue
            turn = bike.with_params(yB=w).turn(phi, delta)
            if turn is not None:
                V[i, j], W[i, j] = turn.V, w
    return V, W


def main():
    args = options(__doc__)
    ds = 1e-2 if args.quick else 4e-3
    bike = Whipple(BENCHMARK)
    torque = st.tongue(bike, ds_max=ds)
    offset = st.tongue(bike, plant="offset", ds_max=ds)
    tip = st.trimmed_turn(bike, *offset.bt, "offset")
    lean = st.largest_stable(bike, offset, lambda x: x.phi, "offset")
    save_json("franke_offset", {
        "upright stable range (m/s)": [offset.v_weave, offset.v_capsize],
        "largest stable lean (deg)": np.degrees(lean.phi),
        "largest stable lean at V (m/s)": lean.V,
        "lowest stable speed (m/s)": tip.V,
        "lowest stable speed at lean (deg)": np.degrees(tip.phi),
        "offset W at lowest stable speed (mm)": 1e3 * st.offset_trim(bike, *offset.bt),
    })  # fmt: skip

    n = 30 if args.quick else 70
    phis = np.radians(np.linspace(1, 16, n))
    deltas = np.geomspace(5e-3, 0.6, 2 * n)
    V, W = offset_grid(bike, phis, deltas)
    lean_grid = np.degrees(phis)[:, None] * np.ones_like(V)
    W = np.where((V < 3.0) & (lean_grid > 11), np.nan, W)  # large offsets far outside the plot
    fig, ax = ps.figure(ps.HALF, 5.6 * ps.CM)
    ps.draw_tongue(ax, bike, torque, beyond=False)
    ps.draw_tongue(ax, bike, offset, plant="offset", color="k", ls=ps.DASH, fill=False,
                   beyond=False)  # fmt: skip
    cs = ax.contour(V, lean_grid, 1e3 * W, levels=[2, 5, 10, 20], colors=ps.CONTOUR,
                    linewidths=0.5, zorder=0)  # fmt: skip
    ps.contour_labels(ax, cs, 14.5, x_min=2.5, x_max=6.3)
    ax.set(xlim=(2.4, 6.4), ylim=(0, 16), xticks=[3, 4, 5, 6], yticks=[0, 5, 10, 15],
           xlabel=ps.XV, ylabel=ps.YLEAN)  # fmt: skip
    ps.label(ax, 3.55, 3.0, "Hopf", color=ps.HOPF, ha="right")
    ps.label(ax, 6.12, 3.0, "Fold", color=ps.FOLD)
    ps.label(ax, 5.1, 6.0, "Torque", ha="center", color=ps.STABLE_TEXT)
    ps.label(ax, 3.0, 11.7, "Offset", ha="right")
    ps.save(fig, "franke_offset")


if __name__ == "__main__":
    main()
