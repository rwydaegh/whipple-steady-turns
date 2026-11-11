"""Turn entry of the benchmark bicycle under a virtual rider, from straight running at 5 m/s into a
right turn of 10 m radius (Appendix D, Fig. 10)."""

import numpy as np

from whipple import plotstyle as ps
from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.output import options, save_json
from whipple.rider import Rider
from whipple.stability import newton_system

V, R, T_RAMP, T_END = 5.0, 10.0, 1.0, 12.0


def steady_turn(bike):
    """Steady turn with rear-contact speed V and radius R."""
    phi = np.arctan(V**2 / (ps.G * R))
    x0 = np.array([phi, bike.p["w"] / (R * np.cos(bike.p["lam"]))])
    x = newton_system(lambda x: (lambda t: np.array([t.V - V, t.R - R]))(bike.turn(*x)), x0)
    return bike.turn(*x)


def main():
    options(__doc__)
    bike = Whipple(BENCHMARK)
    t, Y, T = Rider(bike, V, R, T_RAMP).run(T_END, 2401)
    turn = steady_turn(bike)
    j = int(np.argmin(Y[:, 1]))
    save_json("turn_entry", {
        "smallest steer (deg)": np.degrees(Y[j, 1]),
        "at time (s)": t[j],
        "smallest steer torque (N m)": T.min(),
        "largest lean (deg)": np.degrees(Y[:, 0].max()),
        "final lean, steer (deg)": [np.degrees(Y[-1, 0]), np.degrees(Y[-1, 1])],
        "final steer torque (N m)": T[-1],
        "steady turn lean, steer (deg)": [np.degrees(turn.phi), np.degrees(turn.delta)],
        "steady turn steer torque (N m)": turn.T,
    })  # fmt: skip

    fig, axs = ps.figure(ps.FULL, 8.0 * ps.CM, 3, 1, sharex=True)
    fig.subplots_adjust(hspace=0.18)
    rows = [
        (np.degrees(Y[:, 1]), np.degrees(turn.delta), r"$\delta$ (deg)", [0, 2, 4, 6], (-1.2, 7)),
        (np.degrees(Y[:, 0]), np.degrees(turn.phi), r"$\phi$ (deg)", [0, 5, 10, 15], (-1, 16)),
        (T, turn.T, r"$T$ (N m)", [-0.6, -0.3, 0], (-0.7, 0.1)),
    ]
    for ax, (y, steady, name, ticks, lim) in zip(axs, rows):
        ax.axhline(steady, color=ps.GREY, lw=ps.LW_THIN, ls=ps.DASH, zorder=0)
        ax.axhline(0, color=ps.LIGHT, lw=ps.LW_THIN, zorder=0)
        ax.plot(t, y, color="k")
        ax.set(ylabel=name, yticks=ticks, ylim=lim)
        ax.set_yticklabels([f"${x:g}$" for x in ticks])
    axs[-1].set(xlim=(0, 6), xlabel=r"Time $t$ (s)")
    line = dict(arrowstyle="-", lw=0.5, shrinkA=1, shrinkB=2)
    axs[0].annotate("Countersteer", (t[j], np.degrees(Y[j, 1])), xytext=(1.3, -0.75), fontsize=7,
                    va="center", arrowprops=line)  # fmt: skip
    ps.label(axs[0], 5.95, np.degrees(turn.delta) - 0.4, "Steady turn", color=ps.GREY, ha="right",
             va="top", fontsize=7)  # fmt: skip
    fig.align_ylabels(axs)
    ps.save(fig, "turn_entry")


if __name__ == "__main__":
    main()
