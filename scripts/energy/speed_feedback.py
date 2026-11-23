"""Largest stable lean with speed feedback (not in the paper)."""

import argparse
import math
import time

import numpy as np
from common import RESULTS, bench, quartics, quintic, save, turn_at, turn_state

from whipple.model import complex_step

GAMMAS = [0.0, 0.003, 0.01, 0.03, 0.05, 0.1, 0.11, 0.113, 0.115, 0.12, 0.15, 0.2, 1.0, math.inf]


def grid(bike, lin, Vs, phis):
    nV, nP = len(Vs), len(phis)
    aH, aS = np.full((nV, nP, 5), np.nan), np.full((nV, nP, 5), np.nan)
    gpg = np.full((nV, nP), np.nan)
    t0 = time.time()
    for i, V in enumerate(Vs):
        for j, phi in enumerate(phis):
            try:
                t = turn_at(bike, lin, phi, V)
            except (ValueError, RuntimeError):
                continue
            aH[i, j], aS[i, j], gpg[i, j] = quartics(bike, t)
        print(f"V = {V:.3f} done, {time.time() - t0:.0f} s", flush=True)
    return aH, aS, gpg


def margin(aH, aS, gamma):
    c = aH if gamma == 0 else aS if gamma == math.inf else quintic(aH, aS, gamma)
    return np.max(np.roots(c).real)


def largest_lean(Vs, phis_deg, aH, aS, gamma):
    """Largest lean of the stable set connected to the smallest lean of the grid, over V."""
    ub = np.full(len(Vs), np.nan)
    for i in range(len(Vs)):
        m = np.array([margin(aH[i, j], aS[i, j], gamma) for j in range(len(phis_deg))])
        if not m[0] < 0:
            continue
        k = np.flatnonzero(~(m < 0))
        if len(k):
            k = k[0]
            ub[i] = phis_deg[k - 1] + (phis_deg[k] - phis_deg[k - 1]) * m[k - 1] / (m[k - 1] - m[k])
    i = int(np.nanargmax(ub))
    if 0 < i < len(ub) - 1 and np.all(np.isfinite(ub[i - 1 : i + 2])):
        c = np.polyfit(Vs[i - 1 : i + 2], ub[i - 1 : i + 2], 2)
        v = -c[1] / (2 * c[0])
        return float(np.polyval(c, v)), float(v)
    return float(ub[i]), float(Vs[i])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="grid 0.1 m/s x 0.1 deg")
    quick = p.parse_args().quick
    bike, lin = bench()
    dV, dphi = (0.1, 0.1) if quick else (0.02, 0.02)
    Vs = np.arange(3.8, 5.4 + 1e-9, dV)
    phis_deg = np.arange(10.0, 16.0 + 1e-9, dphi)
    aH, aS, gpg = grid(bike, lin, Vs, np.radians(phis_deg))
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / f"feedback_grid_{dV:g}.npz", Vs=Vs, phis_deg=phis_deg, aH=aH, aS=aS, gpg=gpg)
    rows = []
    for g in GAMMAS:
        L, VL = largest_lean(Vs, phis_deg, aH, aS, g)
        rows.append(dict(gamma=g, largest_lean_deg=L, V=VL))
        print(rows[-1], flush=True)
    cases = {}
    for name, V, phi in (("inside", 5.0, 8.0), ("between the folds", 4.6, 13.0)):
        t = turn_at(bike, lin, math.radians(phi), V)
        h, s, gg = quartics(bike, t)
        z = turn_state(t)
        eig = {}
        for g in (0.0, 0.01, 0.03, 0.1, 1.0):
            lam = np.roots(quintic(h, s, g)) if g > 0 else np.r_[np.roots(h), 0.0]
            eig[str(g)] = sorted(lam, key=lambda x: -x.real)
        lo, hi = 1e-6, 1e3
        crit = None
        if margin(h, s, lo) > 0 > margin(h, s, hi):
            for _ in range(80):
                mid = math.sqrt(lo * hi)
                lo, hi = (lo, mid) if margin(h, s, mid) < 0 else (mid, hi)
            crit = hi
        G = 0.03 / gg
        J = complex_step(lambda x, G=G, t=t: _feedback(bike, t, G)(x), z)
        cases[name] = dict(
            V=V,
            lean_deg=phi,
            T=t.T,
            eigenvalues=eig,
            gamma_crit=crit,
            gamma_crit_small_gamma=-h[4] / (s[3] - h[2] * s[4] / h[3]),
            slow_root_per_gamma=-s[4] / h[4],
            jacobian_eigenvalues_gamma_0p03=sorted(np.linalg.eigvals(J), key=lambda x: -x.real),
        )
        print(name, cases[name]["gamma_crit"], flush=True)
    save(
        "speed_feedback" + ("_quick" if quick else ""),
        dict(
            grid=dict(dV=dV, dphi_deg=dphi),
            largest_lean=rows,
            cases=cases,
            m_eff=float(1 / np.nanmedian(gpg)),
        ),
    )


def _feedback(bike, t, G):
    def f(x):
        _, ud = bike.accelerations(x[0], x[1], x[2:5], t.T, G * (x[4] - t.s))
        return np.r_[x[2:4], ud]

    return f


if __name__ == "__main__":
    main()
