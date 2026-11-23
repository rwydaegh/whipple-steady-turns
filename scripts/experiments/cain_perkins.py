"""Comparison with Cain and Perkins (2012) (Sect. 15.2, Fig. cain, Table cain)."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

from whipple import stability as st
from whipple.bicycles import measured_sets
from whipple.linear import Linear
from whipple.model import Whipple
from whipple.output import _plain

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "experiments"
G = 9.81

# Cain and Perkins (2012), Table A1 (z positive down).
A1_COMMON = dict(
    w=1.060,
    lam=0.3307,
    c=0.064,
    rR=0.322,
    mR=2.420,
    IRyy=0.127,
    rF=0.326,
    mF=1.885,
    IFyy=0.128,
    mH=2.442,
    xH=0.890,
    zH=-0.765,
)
A1_SUBJECT = {1: dict(mT=85.400, xT=0.383, zT=-0.996), 2: dict(mT=89.400, xT=0.355, zT=-1.053)}
A1_K = {
    1: dict(K0=[[-85.050, -2.028], [-2.028, -0.659]], K2=[[0, 76.606], [0, 1.924]]),
    2: dict(K0=[[-94.156, -1.969], [-1.969, -0.639]], K2=[[0, 84.733], [0, 1.872]]),
}
# Cain and Perkins (2012), Table 3: fits T/delta = m V^2 + b (m in kg, b in N m).
TABLE3 = {
    1: {
        "model": (0.097, -5.986),
        "normal": (-0.088, -0.374),
        "lean in": (-0.058, 16.490),
        "lean out": (-0.051, -17.238),
    },
    2: {
        "model": (0.100, -5.868),
        "normal": (0.159, -9.781),
        "lean in": (0.117, 26.432),
        "lean out": (0.127, -36.509),
    },
}
# Cain and Perkins (2012), Table 2: change of the frame roll angle by upper-body lean (deg).
TABLE2_ROLL = {1: {"lean in": -1.3, "lean out": 1.4}, 2: {"lean in": -3.0, "lean out": 2.8}}
RADII = [9.14, 12.19, 18.29, 22.86, 27.74, 28.35, 32.31, 32.61, 32.92]
DONORS = [
    "Browser+Jason",
    "Browserins+Jason",
    "Stratos+Jason",
    "Rigid+Jason",
    "Crescendo+Jason",
    "Rigidcl+Luke",
    "Rigidcl+Charlie",
]


def table_a1_K(s):
    """K0 and K2 from Table A1 by Eqs. A5-A19 of Cain and Perkins (2012)."""
    a, b = A1_COMMON, A1_SUBJECT[s]
    lam = a["lam"]
    mA = a["mH"] + a["mF"]
    xA = (a["xH"] * a["mH"] + a["w"] * a["mF"]) / mA
    zA = (a["zH"] * a["mH"] - a["rF"] * a["mF"]) / mA
    uA = (xA - a["w"] - a["c"]) * math.cos(lam) - zA * math.sin(lam)
    mu = a["c"] * math.cos(lam) / a["w"]
    SF = a["IFyy"] / a["rF"]
    ST = a["IRyy"] / a["rR"] + SF
    SA = mA * uA + mu * b["mT"] * b["xT"]
    K0 = np.array([[b["mT"] * b["zT"], -SA], [-SA, -SA * math.sin(lam)]])
    K2 = np.array(
        [
            [0.0, (ST - b["mT"] * b["zT"]) * math.cos(lam) / a["w"]],
            [0.0, (SA + SF * math.sin(lam)) * math.cos(lam) / a["w"]],
        ]
    )
    return K0, K2


def zero_speed(m, b):
    return math.sqrt(-b / m) if m * b < 0 else None


def parameter_set(s, donor):
    """Whipple set that matches Table A1; frame inertias (scaled by mass) and wheel I_xx (scaled by
    I_yy) from a measured bicycle and rider set."""
    p0 = measured_sets("rider")[donor]
    a, t = A1_COMMON, A1_SUBJECT[s]
    p = dict(p0)
    p.update(a)
    p["IRxx"] = p0["IRxx"] * a["IRyy"] / p0["IRyy"]
    p["IFxx"] = p0["IFxx"] * a["IFyy"] / p0["IFyy"]
    for k in ("IHxx", "IHyy", "IHzz", "IHxz"):
        p[k] = p0[k] * a["mH"] / p0["mH"]
    mB = t["mT"] - a["mR"] - a["mF"] - a["mH"]
    for k in ("IBxx", "IByy", "IBzz", "IBxz"):
        p[k] = p0[k] * mB / p0["mB"]
    p["mB"] = mB
    p["xB"] = (t["mT"] * t["xT"] - a["mH"] * a["xH"] - a["mF"] * a["w"]) / mB
    p["zB"] = (t["mT"] * t["zT"] + a["mR"] * a["rR"] + a["mF"] * a["rF"] - a["mH"] * a["zH"]) / mB
    return p


def linear_ratio(p):
    """T/delta = det K / K_phiphi = m V^2 + b of the linear model, v_c and v_w."""
    lin = Linear(p)

    def ratio(v):
        K = lin.K(v)
        return np.linalg.det(K) / K[0, 0]

    b = ratio(0.0)
    m = ratio(1.0) - b
    return dict(m=m, b=b, v_c=lin.capsize_speed(), v_w=lin.weave_speed((2.0, 9.0)))


def turn_at(bike, V, R, guess):
    def F(x):
        t = bike.turn(*x)
        return [1e3, 1e3] if t is None else [t.V - V, (t.R - R) / R]

    x, _, ier, _ = fsolve(F, guess, full_output=True, xtol=1e-12)
    t = bike.turn(*x)
    return t if ier == 1 and t is not None and abs(t.V - V) < 1e-6 else None


def stable(bike, t, plant):
    ev = st.eigenvalues(bike, t.phi, t.delta, plant)
    return bool(np.all(ev[np.abs(ev) > 1e-8].real < 0))


def nonlinear_rows(p, v2_grid):
    bike = Whipple(p)
    rows = []
    for R in RADII:
        guess = None
        for v2 in v2_grid:
            if guess is None:
                guess = [math.atan(v2 / R / G), p["w"] / (R * math.cos(p["lam"]))]
            t = turn_at(bike, math.sqrt(v2), R, guess)
            if t is None:
                guess = None
                continue
            guess = [t.phi, t.delta]
            rows.append(
                dict(
                    R=R,
                    v2=v2,
                    lean_deg=math.degrees(t.phi),
                    delta=t.delta,
                    ratio=t.T / t.delta,
                    stable_free=stable(bike, t, "free"),
                    stable_constant=stable(bike, t, "constant"),
                )
            )
    return rows


def fit(rows):
    m, b = np.polyfit([r["v2"] for r in rows], [r["ratio"] for r in rows], 1)
    return dict(m=m, b=b, v_zero=zero_speed(m, b))


def digitised():
    pts = {1: [], 2: []}
    with open(DATA / "cain_perkins_2012_fig11.csv") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            pts[int(r["subject"])].append(
                (float(r["v2_m2_per_s2"]), float(r["T_over_delta_Nm_per_rad"]))
            )
    return {s: np.array(v) for s, v in pts.items()}


def bootstrap(pts, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for s, xy in pts.items():
        x, y = xy[:, 0], xy[:, 1]
        m0, b0 = np.polyfit(x, y, 1)
        ms, zs = [], []
        for _ in range(n):
            i = rng.integers(0, len(x), len(x))
            if np.ptp(x[i]) < 1:
                continue
            m, b = np.polyfit(x[i], y[i], 1)
            ms.append(m)
            zs.append(zero_speed(m, b) if m > 0 else np.nan)
        zs = np.array(zs, float)
        out[s] = dict(
            n_points=len(x),
            m=m0,
            b=b0,
            v_zero=zero_speed(m0, b0),
            m_95=np.percentile(ms, [2.5, 97.5]),
            fraction_positive_slope=np.mean(np.array(ms) > 0),
            v_zero_95=np.nanpercentile(zs, [2.5, 97.5]),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="one donor set, coarser speed grid")
    quick = ap.parse_args().quick
    donors = DONORS[3:4] if quick else DONORS
    v2_grid = np.arange(2.0, 92.01, 6.0 if quick else 2.0)
    out = dict(table_a1={}, linear={}, nonlinear={}, band={}, rider_lean={})
    for s in (1, 2):
        K0, K2 = table_a1_K(s)
        m = -(K2[0, 1] * K0[0, 1] - K0[0, 0] * K2[1, 1]) / K0[0, 0]
        b = -G * (K0[0, 1] ** 2 - K0[1, 1] * K0[0, 0]) / K0[0, 0]
        out["table_a1"][s] = dict(
            K0=K0,
            K2=K2,
            K0_printed=A1_K[s]["K0"],
            K2_printed=A1_K[s]["K2"],
            m=m,
            b=b,
            v_zero=zero_speed(m, b),
        )
        delta = A1_COMMON["w"] / (24.0 * math.cos(A1_COMMON["lam"]))
        per_deg = G * K0[1, 0] * math.radians(1.0) / delta
        out["rider_lean"][s] = {
            k: dict(
                roll_change_deg=d,
                predicted_shift=per_deg * d,
                measured_shift=TABLE3[s][k][1] - TABLE3[s]["normal"][1],
            )
            for k, d in TABLE2_ROLL[s].items()
        }
        allrows = []
        for dn in donors:
            p = parameter_set(s, dn)
            L = linear_ratio(p)
            rows = nonlinear_rows(p, v2_grid)
            allrows += rows
            fast = [r["ratio"] for r in rows if 66 <= r["v2"] <= 74]
            out["linear"][f"{s}|{dn}"] = L
            out["nonlinear"][f"{s}|{dn}"] = dict(
                fit=fit(rows),
                per_radius={str(R): fit([r for r in rows if r["R"] == R]) for R in RADII},
                ratio_v2_66_to_74=[min(fast), max(fast)] if fast else None,
                stable_free_v2={
                    str(R): [r["v2"] for r in rows if r["R"] == R and r["stable_free"]]
                    for R in RADII
                },
            )
            print(
                s,
                dn,
                {k: round(v, 4) for k, v in L.items()},
                out["nonlinear"][f"{s}|{dn}"]["fit"],
                flush=True,
            )
        v2s = sorted({r["v2"] for r in allrows})
        out["band"][s] = dict(
            v2=v2s,
            lo=[min(r["ratio"] for r in allrows if r["v2"] == v) for v in v2s],
            hi=[max(r["ratio"] for r in allrows if r["v2"] == v) for v in v2s],
        )
    out["table3"] = TABLE3
    out["bootstrap"] = bootstrap(digitised())
    print(out["bootstrap"], out["rider_lean"])
    import json

    name = "cain_perkins" + ("_quick" if quick else "") + ".json"
    (DATA / name).write_text(json.dumps(_plain(out), indent=1) + "\n")
    print("wrote", (DATA / name).relative_to(REPO))


if __name__ == "__main__":
    main()
