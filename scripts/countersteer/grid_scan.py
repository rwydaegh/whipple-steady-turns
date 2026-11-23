"""Speed-lean grids for thm:countersteer (Sect. 7, "Countersteer in steady turns")."""

import argparse
import json
import math
import time

import numpy as np
from common import (
    RAW,
    WIDE,
    Whipple,
    a4_both,
    all_sets,
    delta_at,
    line_scan,
    solve_decreasing,
    stable,
    trim,
    upright,
)
from scipy.optimize import brentq, minimize_scalar


def H_of(bike, phi, d):
    try:
        v = trim(bike, phi, d)
    except (RuntimeError, np.linalg.LinAlgError):
        return math.nan
    return v[2] if v[0] > 0 else math.nan


def h_path(bike, phi1, d1, H1, n=8):
    """Turns on the level curve H = H1 at the leans phi1 k/n, k = n-1 .. 1."""
    pts, phi_prev, d_prev = [], phi1, d1
    for k in range(n - 1, 0, -1):
        phi = phi1 * k / n
        d = solve_decreasing(lambda x, phi=phi: H_of(bike, phi, x) - H1, d_prev * phi / phi_prev)
        if not np.isfinite(d):
            pts.append(None)
            break
        a = a4_both(bike, phi, d)
        pts.append(dict(T=a["T"], a4=a["a4_free"], H_S=a["H_S"], Rpd=a["R"][0, 1]))
        phi_prev, d_prev = phi, d
    return pts


def first_fold(rec, key, f, sg):
    vals = np.array([r[key] for r in rec])
    k = np.flatnonzero(np.sign(vals) != sg)
    if not len(k):
        return None
    k = int(k[0])
    try:
        return brentq(f, rec[k - 1]["phi"] if k > 0 else 1e-3, rec[k]["phi"], xtol=1e-12)
    except ValueError:
        return None


def scan(name, p, nV=24, phi_max=45.0):
    t0 = time.time()
    bike = Whipple(p)
    u = upright(p)
    lin, vc, vw = u["lin"], u["v_c"], u["v_w"]
    Vs = np.linspace(0.5 * vw if vw == vw else 1.5, 1.35 * vc if vc == vc else 14.0, nV)
    phis = np.radians(np.arange(1.0, phi_max + 1e-9, 1.0))
    U0, M0 = bike.potential(0.0, 0.0), bike.mass_matrix(0.0, 0.0)
    lines, points = [], []
    for V in Vs:
        K = lin.K(V)
        a40 = np.linalg.det(K) / np.linalg.det(lin.M)
        rec = []
        for phi, d in zip(phis, line_scan(bike, lin, V, phis)):
            if not np.isfinite(d):
                break
            a = a4_both(bike, phi, d)
            rec.append(
                dict(
                    V=V,
                    phi=phi,
                    delta=d,
                    T=a["T"],
                    H=a["H"],
                    a4_cs=a["a4_constant"],
                    a4_fr=a["a4_free"],
                    st_cs=stable(bike, phi, d, "constant"),
                    st_fr=stable(bike, phi, d, "free"),
                    Rpd=a["R"][0, 1],
                    Fcphi=a["Fcphi"],
                    H_S=a["H_S"],
                    H_phi_S=a["H_phi_S"],
                    T_S=a["T_S"],
                    T_phi_S=a["T_phi_S"],
                )
            )
        line = dict(V=V, a40=a40, tau1=lin.tau1(V), n=len(rec))

        def at(phi, V=V):
            return a4_both(bike, phi, delta_at(bike, lin, phi, V))

        sg = np.sign(a40)
        line["fold_cs"] = first_fold(rec, "a4_cs", lambda x: at(x)["a4_constant"], sg)
        line["fold_fr"] = first_fold(rec, "a4_fr", lambda x: at(x)["a4_free"], sg)
        fc = line["fold_cs"]
        if fc is not None:
            sgn = 1.0 if a40 > 0 else -1.0
            res = minimize_scalar(
                lambda x, s=sgn, V=V: s * trim(bike, x, delta_at(bike, lin, x, V))[1],
                bounds=(max(fc - 0.03, 1e-3), fc + 0.03),
                method="bounded",
                options=dict(xatol=1e-11),
            )
            a = at(fc)
            line.update(
                T_ext_phi=float(res.x),
                T_fold=a["T"],
                T_S_fold=a["T_S"],
                H_phi_S_fold=a["H_phi_S"],
                a4_fr_at_cs_fold=a["a4_free"],
            )
        lines.append(line)
        for r in rec:
            r["below_first_fold_cs"] = bool(fc is None or r["phi"] < fc)
            points.append(r)
    st_fr = [r for r in points if r["st_fr"]]
    for r in st_fr[:: max(1, len(st_fr) // 150)]:
        path = h_path(bike, r["phi"], r["delta"], r["H"])
        ok = all(q is not None for q in path)
        r["path_complete"] = ok
        if ok and path:
            Ts = [r["T"]] + [q["T"] for q in path] + [0.0]
            r["path_min_a4"] = min(q["a4"] for q in path)
            r["path_T_monotone"] = bool(np.all(np.diff(Ts) > 0) or np.all(np.diff(Ts) < 0))
        r["v_H"] = math.sqrt(2 * (r["H"] - U0) / (M0[2, 2] / p["rR"] ** 2))
    return dict(name=name, v_c=vc, v_w=vw, lines=lines, points=points, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="parameter sets (default: all 42)")
    ap.add_argument("--quick", action="store_true", help="benchmark only")
    args = ap.parse_args()
    sets = all_sets()
    names = ["benchmark"] if args.quick else (args.names or list(sets))
    RAW.mkdir(parents=True, exist_ok=True)
    for name in names:
        o = scan(name, sets[name], phi_max=60.0 if name in WIDE else 45.0)
        (RAW / f"{name}.json").write_text(json.dumps(o, default=float))
        print(f"{name}: {len(o['points'])} turns, {o['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
