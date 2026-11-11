"""Eigenvalues of steady turns of the benchmark bicycle over speed at four leans (Fig. 2)."""

import numpy as np
from scipy.optimize import brentq, linear_sum_assignment

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.bicycles import BENCHMARK
from whipple.linear import Linear
from whipple.model import Whipple
from whipple.output import options

LEANS = (0, 5, 10, 15)
V_MAX = 10.0


def track(L):
    """Order the eigenvalue columns continuously from row to row."""
    L = L.copy()
    for i in range(1, len(L)):
        _, cols = linear_sum_assignment(np.abs(L[i - 1][:, None] - L[i][None, :]))
        L[i] = L[i][cols]
    return L


def sweep(bike, lin, lean_deg, n):
    """Speeds and eigenvalues of the turns at one lean, from V_MAX down."""
    if lean_deg == 0:
        V = np.linspace(V_MAX, 0.0, n)
        return V, np.array([lin.eigenvalues(v) for v in V])
    phi = np.radians(lean_deg)
    d0 = brentq(lambda d: bike.turn(phi, d).V - V_MAX, 1e-6, 0.5, xtol=1e-15)
    V, L = [], []
    for d in d0 * np.geomspace(1, 1.2 / d0, n):
        turn = bike.turn(phi, d)
        if turn is None or (V and turn.V >= V[-1]):
            break
        V.append(turn.V)
        L.append(st.eigenvalues(bike, phi, d))
    return np.array(V), np.array(L)


def names(L):
    """Mode names at the high-speed end: weave is the complex pair, castor the most negative."""
    real = [k for k in range(4) if abs(L[k].imag) < 1e-9]
    castor = min(real, key=lambda k: L[k].real)
    return ["Weave" if k not in real else "Castor" if k == castor else "Capsize" for k in range(4)]


def main():
    args = options(__doc__)
    bike, lin = Whipple(BENCHMARK), Linear(BENCHMARK)
    n = 300 if args.quick else 1500
    colors = {"Weave": ps.HOPF, "Capsize": ps.FOLD, "Castor": ps.CASTOR}
    fig, axs = ps.figure(ps.FULL, 9.6 * ps.CM, 2, 2, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.10, hspace=0.28)
    for ax, lean, letter in zip(axs.flat, LEANS, "abcd"):
        V, L = sweep(bike, lin, lean, n)
        L = track(L)
        stable = np.all(L.real < 0, axis=1)
        ax.fill_between(V, -10, 10, where=stable, color=ps.STABLE, alpha=ps.STABLE_ALPHA, lw=0)
        ax.axhline(0, color=ps.LIGHT, lw=ps.LW_THIN, zorder=0.5)
        mode = names(L[0])
        for k in range(4):
            ax.plot(V, L[:, k].real, color=colors[mode[k]])
        k = mode.index("Weave")
        im = np.abs(L[:, k].imag)
        ax.plot(V, np.where(im > 1e-6, im, np.nan), color=ps.HOPF, lw=ps.LW_THIN, ls=ps.DASH)
        ax.set(xlim=(0, V_MAX), ylim=(-10, 10), xticks=range(0, 11, 2), yticks=[-10, -5, 0, 5, 10])
        ax.text(0.97, 0.05, rf"$\phi={lean}^\circ$", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8)  # fmt: skip
        ps.panel(ax, letter, x=-0.13)
    for ax in axs[1]:
        ax.set_xlabel(ps.XV)
    for ax in axs[:, 0]:
        ax.set_ylabel("Eigenvalue (1/s)")
    a = axs[0, 0]
    ps.label(a, 8.6, 7.0, r"$\mathrm{Im}\,\lambda_w$", color=ps.HOPF, ha="center", va="top")
    ps.label(a, 8.5, -3.9, "Weave", color=ps.HOPF, ha="center", va="top")
    ps.label(a, 8.5, 1.4, "Capsize", color=ps.FOLD, ha="center")
    ps.label(a, 3.1, -8.4, "Castor", color=ps.CASTOR)
    ps.save(fig, "eigenvalues_vs_speed")


if __name__ == "__main__":
    main()
