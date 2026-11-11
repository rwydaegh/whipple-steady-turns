"""First Lyapunov coefficient l1 of the weave Hopf bifurcation, upright and along the cornering Hopf
curve, and the Bautin point of bicycle B where l1 changes sign (Sect. 7.3, Fig. 8)."""

import numpy as np
from matplotlib.ticker import FixedLocator
from scipy.optimize import brentq, newton

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.bicycles import BICYCLES
from whipple.linear import Linear
from whipple.model import Whipple
from whipple.output import options, save_json
from whipple.weave import first_lyapunov, leaf_at_turn, upright_leaf


def l1_at(bike, x):
    return first_lyapunov(*leaf_at_turn(bike, *x))


def along_hopf(bike, t):
    """Lean (deg) and l1 along the Hopf curve, from upright to its largest lean."""
    curve = t.hopf[: int(np.argmax(t.hopf[:, 0])) + 1]
    return np.degrees(curve[:, 0]), np.array([l1_at(bike, x)[0] for x in curve])


def hopf_steer(bike, phi, guess):
    return newton(lambda d: st.hurwitz(st.characteristic(bike, phi, d))[0], guess, tol=1e-14)


def bautin(bike, t, lean_deg, l1):
    """Hopf point where l1 = 0, from the first sign change along the curve."""
    k = int(np.flatnonzero(np.diff(np.sign(l1)))[0])
    ratio = t.hopf[k, 1] / t.hopf[k, 0]

    def l1_of(phi):
        return l1_at(bike, (phi, hopf_steer(bike, phi, ratio * phi)))[0]

    phi = brentq(l1_of, *np.radians(lean_deg[k : k + 2]), xtol=1e-12)
    delta = hopf_steer(bike, phi, ratio * phi)
    turn = bike.turn(phi, delta)
    return {"lean (rad)": phi, "lean (deg)": np.degrees(phi), "speed V (m/s)": turn.V,
            "radius R (m)": turn.R, "steer torque (N m)": turn.T,
            "weave frequency (rad/s)": l1_at(bike, (phi, delta))[1]}  # fmt: skip


def figure(curves, root):
    fig, axs = ps.figure(ps.FULL, 6.0 * ps.CM, 1, 2)
    fig.subplots_adjust(wspace=0.36)
    ax = axs[0]
    for name, (lean, l1) in curves.items():
        m = l1 < 0
        ax.plot(lean[m], -l1[m], color=ps.BIKE[name])
    ticks = [0.001, 0.01, 0.1, 1, 10, 100]
    ax.set(yscale="log", ylim=(1e-3, 200), xlim=(0, 14), xticks=[0, 4, 8, 12])
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_yticklabels([f"${x:g}$" for x in ticks])
    ax.set(xlabel=ps.YLEAN, ylabel=r"$-\ell_1$")
    ps.label(ax, 0.5, 2.6, "Benchmark", va="bottom")
    ps.label(ax, 1.0, 0.045, "A", color=ps.BIKE["A"], va="bottom")
    ps.label(ax, 5.6, 0.0016, "B", color=ps.BIKE["B"])

    ax = axs[1]
    for name in ("A", "B"):
        ax.plot(*curves[name], color=ps.BIKE[name])
    ax.axhline(0, color=ps.LIGHT, lw=ps.LW_THIN, zorder=0)
    ps.bt_marker(ax, root["lean (deg)"], 0, color=ps.BIKE["B"])
    ax.annotate(rf"Bautin, ${root['lean (deg)']:.2f}^\circ$", (root["lean (deg)"], 0),
                xytext=(0.6, -0.013), fontsize=7, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, shrinkB=2.5, shrinkA=1))  # fmt: skip
    ax.set(xlim=(0, 8), ylim=(-0.07, 0.012), xticks=[0, 2, 4, 6, 8],
           yticks=[-0.06, -0.04, -0.02, 0], xlabel=ps.YLEAN, ylabel=r"$\ell_1$")  # fmt: skip
    ps.label(ax, 2.0, -0.047, "A", color=ps.BIKE["A"], va="top")
    ps.label(ax, 6.7, 0.0005, "B", color=ps.BIKE["B"], va="bottom")
    ps.panel(axs[0], "a", x=-0.27)
    ps.panel(axs[1], "b", x=-0.27)
    ps.save(fig, "lyapunov_coefficient")


def main():
    args = options(__doc__)
    ds = 1e-2 if args.quick else 4e-3
    out, curves = {}, {}
    for name, p in BICYCLES.items():
        bike = Whipple(p)
        l1, omega = first_lyapunov(*upright_leaf(bike, Linear(p).weave_speed()))
        t = st.tongue(bike, ds_max=ds)
        curves[name] = along_hopf(bike, t)
        out[name] = {"upright l1": l1, "upright weave frequency (rad/s)": omega}
    root = bautin(bike, t, *curves["B"])
    out["B"]["Bautin point"] = root
    save_json("lyapunov_coefficient", out)
    figure(curves, root)


if __name__ == "__main__":
    main()
