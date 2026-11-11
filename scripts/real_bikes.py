"""Region of stable turns of ten measured bicycles (parameters of Moore, BicycleParameters):
upright weave and capsize speeds, Bogdanov-Takens tip and largest stable lean. Reproduces the
tongues of measured bicycles (speed-lean plane and normalised plane). Browser, Silver and
Yellowrev have no ordinary tongue and are not computed."""

import json
from pathlib import Path

import numpy as np

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.model import Whipple
from whipple.output import options, save_json

DATA = Path(__file__).resolve().parent.parent / "whipple" / "data" / "measured_bicycles.json"
COLORS = ["#000000", "#6baed6", "#FF7F00", "#2ca02c", "#d62728", "#8c564b", "#c49c94",
          "#9467bd", "#bcbd22", "#e377c2"]  # fmt: skip


def summary(bike, t):
    bt = bike.turn(*t.bt)
    lean = st.largest_stable(bike, t, lambda x: x.phi)
    return {
        "upright weave speed (m/s)": t.v_weave,
        "upright capsize speed (m/s)": t.v_capsize,
        "BT lean (deg)": np.degrees(bt.phi),
        "BT speed V (m/s)": bt.V,
        "BT steer torque (N m)": bt.T,
        "largest stable lean (deg)": np.degrees(lean.phi),
        "largest stable a_y/g": lean.ay / ps.G,
    }


def main():
    args = options(__doc__)
    ds = 1e-2 if args.quick else 4e-3
    params = json.loads(DATA.read_text())
    fig, axs = ps.figure(ps.FULL, 7.0 * ps.CM, 1, 2)
    fig.subplots_adjust(wspace=0.30)
    out = {}
    for (name, p), color in zip(params.items(), COLORS):
        bike = Whipple(p)
        t = st.tongue(bike, phi_max=0.8, ds_max=ds, weave_bracket=(2.5, 6.5))
        out[name] = summary(bike, t)
        x, y = ps.tongue_curve(bike, np.vstack([t.hopf, t.fold[::-1]]))
        tip = out[name]["BT lean (deg)"]
        vw = t.v_weave
        lw = 1.3 if name == "Benchmark" else 0.8
        axs[0].plot(x, y, color=color, lw=lw, label=name)
        axs[1].plot(x / vw, y / tip, color=color, lw=lw)
        axs[0].plot(out[name]["BT speed V (m/s)"], tip, "o", ms=3, mfc="white", mec=color, mew=0.7)
        axs[1].plot(out[name]["BT speed V (m/s)"] / vw, 1.0, "o", ms=3, mfc="white", mec=color,
                    mew=0.7)  # fmt: skip
    axs[0].set(xlabel=ps.XV, ylabel=ps.YLEAN, xlim=(1.5, 9.0), ylim=(0, 25))
    axs[1].set(xlabel=r"Speed over weave speed $V/v_w$", ylabel=r"Lean over tip lean",
               xlim=(0.5, 2.1), ylim=(0, 1.8))  # fmt: skip
    ps.panel(axs[0], "a")
    ps.panel(axs[1], "b")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.12), fontsize=8)  # fmt: skip
    ps.save(fig, "real_bikes")
    save_json("real_bikes", out)


if __name__ == "__main__":
    main()
