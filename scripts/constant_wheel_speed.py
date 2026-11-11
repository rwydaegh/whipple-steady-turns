"""Turns at constant steer torque with the rear-wheel speed held (Sect. 4.6, Fig. 5, Table 1)."""

import numpy as np

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.bicycles import BICYCLES
from whipple.model import Whipple
from whipple.output import options, save_json


def summary(bike, t):
    bt = bike.turn(*t.bt)
    lean = st.largest_stable(bike, t, lambda x: x.phi, "held")
    ay = st.largest_stable(bike, t, lambda x: x.ay, "held")
    return {
        "BT lean (deg)": np.degrees(bt.phi),
        "BT speed V (m/s)": bt.V,
        "BT steer torque (N m)": bt.T,
        "largest stable lean (deg)": np.degrees(lean.phi),
        "largest stable lean at V (m/s)": lean.V,
        "largest stable a_y/g": ay.ay / ps.G,
        "largest stable a_y at V (m/s)": ay.V,
    }


def main():
    args = options(__doc__)
    ds = 1e-2 if args.quick else 4e-3
    names = ["benchmark"] if args.quick else list(BICYCLES)
    out = {}
    for name in names:
        bike = Whipple(BICYCLES[name])
        free = st.tongue(bike, ds_max=ds)
        held = st.tongue(bike, plant="held", ds_max=ds)
        out[name] = summary(bike, held)
        if name == "benchmark":
            fig, ax = ps.figure(ps.HALF, 5.6 * ps.CM)
            ps.draw_tongue(ax, bike, free, beyond=False)
            ps.draw_tongue(ax, bike, held, ls=ps.DASH, fill=False, beyond=False, filled_marker=True)
            ax.set(xlim=(2.4, 6.4), ylim=(0, 16), xticks=[3, 4, 5, 6], yticks=[0, 5, 10, 15],
                   xlabel=ps.XV, ylabel=ps.YLEAN)  # fmt: skip
            ps.label(ax, 3.55, 3.0, "Hopf", color=ps.HOPF, ha="right")
            ps.label(ax, 6.12, 3.0, "Fold", color=ps.FOLD)
            ps.label(ax, 4.68, 14.6, "Constant wheel speed", ha="center")
            ps.label(ax, 5.1, 6.0, "Free rolling", ha="center", color=ps.STABLE_TEXT)
            ps.save(fig, "constant_wheel_speed")
    save_json("constant_wheel_speed", out)


if __name__ == "__main__":
    main()
