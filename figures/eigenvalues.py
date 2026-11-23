"""Eigenvalues of steady turns of the benchmark bicycle against speed (fig:eigs, Sect. 4,
"Eigenvalues against speed").
"""

import numpy as np
from figstyle import (
    CAPSIZE,
    CM,
    COL_W,
    DASH,
    LIGHT,
    LW,
    LW_THIN,
    STABLE,
    STABLE_ALPHA,
    STABLE_TEXT,
    WEAVE,
    XV,
    label,
    panel,
    plt,
    save_paper,
)
from scipy.optimize import brentq, linear_sum_assignment

from whipple import stability as st
from whipple.bicycles import BENCHMARK
from whipple.linear import Linear
from whipple.model import Whipple

CASTOR = "#000000"
LEANS = (0, 5, 10, 15)
V_MAX = 10.0
N = 700


def track(L):
    """Order the eigenvalues so that each column is one continuous branch."""
    L = L.copy()
    for i in range(1, len(L)):
        _, cols = linear_sum_assignment(np.abs(L[i - 1][:, None] - L[i][None, :]))
        L[i] = L[i][cols]
    return L


def sweep(bike, lin, lean_deg, n):
    """Eigenvalues along the turns of one lean, from 10 m/s down to a steer of 1.2 rad."""
    if lean_deg == 0:
        V = np.r_[np.linspace(V_MAX, 1.0, n), np.linspace(1.0, 0.0, n // 2)[1:]]
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
    real = [k for k in range(4) if abs(L[k].imag) < 1e-9]
    castor = min(real, key=lambda k: L[k].real)
    return ["Weave" if k not in real else "Castor" if k == castor else "Capsize" for k in range(4)]


def compute():
    bike, lin = Whipple(BENCHMARK), Linear(BENCHMARK)
    out = {}
    for lean in LEANS:
        V, L = sweep(bike, lin, lean, N)
        L = track(L)
        out[str(lean)] = dict(
            V=np.round(V, 6).tolist(),
            re=np.round(L.real, 5).tolist(),
            im=np.round(L.imag, 5).tolist(),
            modes=names(L[0]),
        )
    out["upright"] = dict(
        v_w=float(lin.weave_speed()),
        v_c=float(lin.capsize_speed()),
        zero_speed=sorted(np.real(lin.eigenvalues(0.0)).tolist()),
    )
    return out


def bands(V, stable):
    """Speed intervals where all four eigenvalues have negative real parts."""
    out, start = [], None
    order = np.argsort(V)
    V, stable = V[order], stable[order]
    for v, s in zip(V, stable):
        if s and start is None:
            start = v
        if not s and start is not None:
            out.append((start, v))
            start = None
    if start is not None:
        out.append((start, V[-1]))
    return out


COLORS = {"Weave": WEAVE, "Capsize": CAPSIZE, "Castor": CASTOR}


def main():
    data = compute()  # about 15 s
    fig, axs = plt.subplots(2, 2, figsize=(COL_W, 9.2 * CM), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.22)
    saved = {}
    for ax, lean, letter in zip(axs.flat, LEANS, "abcd"):
        d = data[str(lean)]
        V, re, im = np.array(d["V"]), np.array(d["re"]), np.array(d["im"])
        bb = bands(V, np.all(re < 0, axis=1))
        for a, b in bb:
            ax.axvspan(a, b, color=STABLE, alpha=STABLE_ALPHA, lw=0, zorder=0)
        ax.axhline(0, color=LIGHT, lw=LW_THIN, zorder=0.5)
        for k, m in enumerate(d["modes"]):
            ax.plot(
                V,
                re[:, k],
                color=COLORS[m],
                lw=LW,
                zorder=3,
                ls=(0, (5, 1.5, 1, 1.5)) if m == "Castor" else "-",
            )
        wim = np.abs(im[:, d["modes"].index("Weave")])
        ax.plot(V, np.where(wim > 1e-6, wim, np.nan), color=WEAVE, lw=LW_THIN, ls=DASH, zorder=2)
        ax.set(xlim=(0, V_MAX), ylim=(-10, 10), xticks=range(0, 11, 2), yticks=[-10, -5, 0, 5, 10])
        ax.text(
            0.97,
            0.05,
            rf"$\phi={lean}^\circ$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
        panel(ax, letter, x=-0.17 if letter in "ac" else -0.03, y=1.02)
        saved[f"lean_{lean}"] = dict(stable_bands=bb)
    for ax in axs[1]:
        ax.set_xlabel(XV)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Eigenvalue $\lambda$ (1/s)")
    a = axs[0, 0]
    label(a, 8.3, 7.2, r"$|\mathrm{Im}\,\lambda|$", color=WEAVE, ha="center", va="top")
    label(a, 8.3, -3.9, "Weave", color=WEAVE, ha="center", va="top")
    label(a, 8.3, 1.2, "Capsize", color=CAPSIZE, ha="center")
    label(a, 0.35, -9.1, "Castor", color=CASTOR)
    z = data["upright"]["zero_speed"]
    for y in z:
        a.plot(0, y, "o", ms=3.0, mfc="white", mec="k", mew=0.7, zorder=6, clip_on=False)
    a.annotate(
        "Steering collapse",
        xy=(0.05, z[3]),
        xytext=(0.7, 8.6),
        fontsize=7.5,
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=1, shrinkB=2),
    )
    a.annotate(
        "Whole-bicycle fall",
        xy=(0.05, z[2]),
        xytext=(0.3, -1.5),
        fontsize=7.5,
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=1, shrinkB=2, relpos=(0.04, 1.0)),
    )
    a.text(5.16, -9.0, "Stable", color=STABLE_TEXT, fontsize=7.5, ha="center", va="center")
    saved["upright"] = data["upright"]
    saved["curves"] = {k: v for k, v in data.items() if k != "upright"}
    save_paper(fig, "eigenvalues", saved)
    print(data["upright"], {k: v["stable_bands"] for k, v in saved.items() if k.startswith("lean")})


if __name__ == "__main__":
    main()
