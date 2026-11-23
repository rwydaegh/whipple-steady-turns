"""Independent recomputation of the main numbers of the paper and its supplementary tables."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from independent import (
    Bike,
    complex_jacobian,
    fold_identity,
    fold_saddle,
    largest_on_fold,
    pencil_coefficients,
    pencil_matrices,
    power_residual,
    spring_constants_closed,
    spring_constants_elimination,
    tau3,
    tip_on_fold,
    upright_coefficients,
    upright_speeds,
    zero_speed_rates,
)
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from whipple.bicycles import BICYCLES, measured_sets
from whipple.model import PARAMETERS

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "checks" / "independent.json"
G = 9.81
DEG = 180 / math.pi
BENCH = BICYCLES["benchmark"]

# Measured sets, free rolling (Table S "measured-sets"): v_w, v_c, V_a3, tip V, tip lean, lean*, V*.
MEASURED = {
    "Balanceassistv1+Jason": "4.08 6.24 3.10 3.02 7.42 8.98 4.45",
    "Browser+Aurelia": "4.79 6.59 3.44 3.34 10.07 12.07 4.80",
    "Browser+Charlie": "5.04 7.41 3.64 3.50 10.73 13.43 5.37",
    "Browser+Chris": "4.82 6.85 3.49 3.39 10.28 12.48 4.98",
    "Browser+Jason": "5.14 7.72 3.68 3.55 10.98 14.06 5.60",
    "Browser+Jason72(Hess2012)": "5.01 7.15 3.75 3.60 10.90 13.02 5.19",
    "Browser+Luke": "5.13 7.53 3.63 3.50 10.81 13.78 5.48",
    "Browser+Mont": "5.33 8.18 3.81 3.65 11.35 14.86 5.93",
    "Browserins+Jason": "5.25 7.73 3.85 3.69 11.35 14.17 5.63",
    "Crescendo+Jason": "5.58 9.04 4.32 4.19 13.87 17.26 6.50",
    "DelftBrowser+Arend(Moore2012)": "5.10 7.34 3.93 3.73 10.63 12.53 5.31",
    "DelftBrowser+Jason(Moore2012)": "4.76 6.42 3.60 3.46 9.73 11.10 4.67",
    "Fisher+Jason72(Hess2012)": "5.18 8.40 4.12 3.95 12.71 15.56 6.01",
    "Gyro+Child": "3.49 5.95 2.71 2.70 12.74 16.15 4.27",
    "Pista+Jason72(Hess2012)": "4.81 7.73 3.93 3.78 11.55 13.55 5.45",
    "Rigid+Jason": "5.78 10.14 4.21 4.07 12.93 18.31 7.41",
    "Rigid+rider(Schwab2013)": "6.20 10.85 4.57 4.36 14.73 21.33 8.12",
    "Rigidcl+Charlie": "5.67 9.69 4.28 4.11 12.90 17.31 7.02",
    "Rigidcl+Luke": "5.77 9.88 4.27 4.11 13.00 17.91 7.22",
    "Stratos+Jason": "4.75 8.40 3.34 3.37 10.48 14.84 6.07",
    "Yellow+Jason": "4.54 7.05 3.26 3.23 9.09 11.57 5.08",
    "Yellow+Jason72(Hess2012)": "4.33 6.09 3.39 3.29 8.89 10.02 4.39",
    "Yellowrev+Jason": "5.82 14.24 4.13 4.21 16.38 30.40 10.83",
}
QUICK_SETS = [
    "Browser+Jason",
    "Rigid+Jason",
    "Crescendo+Jason",
    "Pista+Jason72(Hess2012)",
    "Balanceassistv1+Jason",
]
MEASURED_COLUMNS = [
    "v_w (m/s)",
    "v_c (m/s)",
    "V_a3 (m/s)",
    "tip V (m/s)",
    "tip lean (deg)",
    "largest stable lean (deg)",
    "at V (m/s)",
]

ROWS = []


def value(where, quantity, paper, independent):
    """Row for a stated number; agrees if the independent value rounds to the paper's digits."""
    digits = len(paper.split(".")[1]) if "." in paper else 0
    diff = float(independent) - float(paper)
    ok = abs(diff) <= 0.5 * 10.0**-digits * (1 + 1e-9) + 1e-12
    ROWS.append(
        dict(
            where=where,
            quantity=quantity,
            paper=paper,
            independent=float(independent),
            difference=diff,
            agrees=bool(ok),
        )
    )


def identity(where, quantity, residual, tol):
    """Row for an exact statement: the residual must be below tol."""
    ROWS.append(
        dict(
            where=where,
            quantity=quantity,
            paper="exact",
            independent=float(residual),
            difference=float(residual),
            agrees=bool(abs(residual) < tol),
            tol=tol,
        )
    )


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(1.0, np.max(np.abs(b)))


# ---- upright model and spring --------------------------------------------------------------------


def upright_and_spring():
    w = "Sect. 5.2 Region of the benchmark bicycle"
    u = upright_speeds(BENCH)
    value(w, "benchmark v_w (m/s)", "4.292", u["v_w"])
    value(w, "benchmark v_c (m/s)", "6.024", u["v_c"])
    value("Sect. 8.3 Tip speed", "benchmark V_a3 (m/s)", "2.761", u["V_a3"])
    r = zero_speed_rates(BENCH)
    value("Sect. 4 Eigenvalues against speed", "benchmark zero-speed rate 1 (1/s)", "3.13", r[0])
    value("Sect. 4 Eigenvalues against speed", "benchmark zero-speed rate 2 (1/s)", "5.53", r[1])
    for name in ("A", "B"):
        ub = upright_speeds(BICYCLES[name])
        value("Table S region", f"{name} v_w (m/s)", "4.805", ub["v_w"])
        value("Table S region", f"{name} v_c (m/s)", "6.105", ub["v_c"])
    closed, elim = spring_constants_closed(BENCH), spring_constants_elimination(BENCH)
    identity(
        "Eq. (Va3)",
        "V_a3 closed form against root of upright a3",
        closed["V_a3"] - u["V_a3"],
        1e-10,
    )
    identity(
        "Eq. (rider-kc)",
        "kappa_c closed form against elimination",
        closed["kappa_c"] - elim["kappa_c"],
        1e-12,
    )
    w = "Sect. 13.2 Centring spring"
    value(w, "benchmark kappa_c (N m s^2/rad/m^2)", "0.1946", elim["kappa_c"])
    value(w, "benchmark k0 (N m/rad)", "7.061", elim["k0"])
    kstar = elim["kappa_c"] * (u["v_c"] ** 2 - u["V_a3"] ** 2)
    value(w, "benchmark k* (N m/rad)", "5.578", kstar)
    # At k* the upright capsize coefficient vanishes at V_a3.
    identity(
        w,
        "upright a4 at V_a3 with k = k*",
        upright_coefficients(BENCH, u["V_a3"], k=kstar)[3],
        1e-9,
    )
    cstar = brentq(lambda c: upright_coefficients(BENCH, u["v_c"], c=c)[2], 1.0, 100.0, xtol=1e-12)
    value("Sect. S rider spring", "benchmark c* (N m s/rad)", "14.17", cstar)
    ks, cs = [], []
    for p in measured_sets("rider").values():
        q = upright_speeds(p)
        e = spring_constants_elimination(p)
        ks.append(e["kappa_c"] * (q["v_c"] ** 2 - q["V_a3"] ** 2))
        cs.append(brentq(lambda c, p=p, q=q: upright_coefficients(p, q["v_c"], c=c)[2], 0.1, 1e3))
    value(w, "measured sets: smallest k* (N m/rad)", "0.98", min(ks))
    value(w, "measured sets: largest k* (N m/rad)", "15.1", max(ks))
    value(w, "measured sets: median k* (N m/rad)", "5.2", statistics.median(ks))
    value("Sect. S rider spring", "measured sets: smallest c* (N m s/rad)", "3.0", min(cs))
    value("Sect. S rider spring", "measured sets: largest c* (N m s/rad)", "144", max(cs))
    value(
        "Sect. S rider spring",
        "measured sets: median c* (N m s/rad)",
        "12.5",
        statistics.median(cs),
    )
    return u, kstar


# ---- regions of stable turns -------------------------------------------------------------------


def region(name, p, u, plant="free", **control):
    """BT point, largest stable lean and largest stable lateral acceleration on the fold."""
    b = Bike(p)
    v_c = upright_speeds(p, k=control.get("k", 0.0))["v_c"]
    (Vbt, phibt), _ = tip_on_fold(b, 0.8 * u["V_a3"], v_c * (1 - 1e-4), plant, **control)
    V, phi, _ = largest_on_fold(b, Vbt, v_c * (1 - 1e-6), plant, **control)
    if V - Vbt < 1e-4:  # the fold has no interior maximum on the stable side: the tip is highest
        V, phi = Vbt, phibt
    return b, (Vbt, phibt), (V, phi), v_c


def regions(u, kstar):
    w = "Sect. 5.2 Region of the benchmark bicycle"
    b, (Vbt, phibt), (V, phi), v_c = region("benchmark", BENCH, u)
    z, T = b.trim(Vbt, phibt)
    turn = b.model.turn(z[0], z[1])
    roots = sorted(np.roots(np.r_[1.0, b.quartic(Vbt, phibt)]), key=abs)[2:]
    value(w, "benchmark BT lean (deg)", "9.26", phibt * DEG)
    value("Sect. 8.3 Tip speed", "benchmark BT speed (m/s)", "2.790", Vbt)
    value(w, "benchmark BT steer torque (N m)", "-0.858", T)
    value("Table S region", "benchmark BT steer (rad)", "0.226", z[1])
    value("Table S region", "benchmark BT radius (m)", "4.60", turn.R)
    value("Table S region", "benchmark BT a_y/g", "0.173", turn.ay / G)
    value("Table S region", "benchmark BT other eigenvalue 1 (1/s)", "-8.50", roots[1].real)
    value("Table S region", "benchmark BT other eigenvalue 2 (1/s)", "-0.408", roots[0].real)
    value(w, "benchmark largest stable lean (deg)", "12.47", phi * DEG)
    value(w, "benchmark largest stable lean at V (m/s)", "4.54", V)
    Vay, _, ay = largest_on_fold(b, Vbt, v_c * (1 - 1e-6), quantity="ay")
    value(w, "benchmark largest stable a_y/g", "0.227", ay / G)
    value("Table S region", "benchmark largest stable a_y at V (m/s)", "4.47", Vay)
    _, _, (_, phis), _ = region("benchmark", BENCH, u, "constant")
    value(
        "Sect. S order of the two folds",
        "benchmark largest stable lean, constant speed (deg)",
        "13.84",
        phis * DEG,
    )
    paper = {
        "A": ("12.04", "3.29", "-0.296", "15.16", "4.81", "0.266", "4.73"),
        "B": ("12.50", "3.34", "-0.294", "16.07", "4.94", "0.282", "4.87"),
    }
    for name, vals in paper.items():
        p = BICYCLES[name]
        b, (Vbt, phibt), (V, phi), v_c = region(name, p, upright_speeds(p))
        Vay, _, ay = largest_on_fold(b, Vbt, v_c * (1 - 1e-6), quantity="ay")
        got = (phibt * DEG, Vbt, b.trim(Vbt, phibt)[1], phi * DEG, V, ay / G, Vay)
        labels = (
            "BT lean (deg)",
            "BT speed (m/s)",
            "BT steer torque (N m)",
            "largest stable lean (deg)",
            "largest stable lean at V (m/s)",
            "largest stable a_y/g",
            "largest stable a_y at V (m/s)",
        )
        for label, pv, iv in zip(labels, vals, got):
            value("Table S region", f"{name} {label}", pv, iv)
    w = "Sect. 13.2 Centring spring"
    for frac, paper_lean in ((0.4, "8.13"), (0.8, "4.04")):
        _, _, (V, phi), _ = region("benchmark", BENCH, u, k=frac * kstar)
        value(w, f"benchmark largest stable lean at k = {frac} k* (deg)", paper_lean, phi * DEG)


def countersteer(n):
    """Largest steer torque over the stable turns of a (V, lean) grid below v_c and 30 deg."""
    b = Bike(BENCH)
    worst, count = -np.inf, 0
    for V in np.linspace(2.8, 6.0, n):
        for phi in np.linspace(0.005, math.radians(30), n):
            try:
                a = b.quartic(V, phi)
            except RuntimeError:  # no forward steady turn
                continue
            if np.max(np.roots(np.r_[1.0, a]).real) < 0:
                count += 1
                worst = max(worst, b.trim(V, phi)[1])
    ROWS.append(
        dict(
            where="Sect. 5.2 Region of the benchmark bicycle",
            quantity=f"largest steer torque of {count} stable grid turns (N m)",
            paper="negative",
            independent=float(worst),
            difference=None,
            agrees=bool(worst < 0),
        )
    )


def critical_trails():
    w = "Table S trails"
    c146 = brentq(
        lambda c: tau3(Bike({**BENCH, "c": c}), upright_speeds({**BENCH, "c": c})["v_c"]),
        0.12,
        0.18,
        xtol=1e-9,
    )
    value(w, "trail where tau3(v_c) changes sign (mm)", "146.2", 1e3 * c146)
    for plant, guess, paper in (
        ("constant", [8.8, 0.58, 0.165], ("162.5", "33.0", "8.57")),
        ("free", [9.3, 0.58, 0.20], ("198.9", "33.3", "9.30")),
    ):
        V, phi, c = fold_saddle(BENCH, plant, guess)
        turn = Bike({**BENCH, "c": c}).turn(V, phi)
        value(w, f"opening trail, {plant} (mm)", paper[0], 1e3 * c)
        value(w, f"lean at opening, {plant} (deg)", paper[1], phi * DEG)
        value(w, f"speed at opening, {plant} (m/s)", paper[2], V)
        value(w, f"a_y/g at opening, {plant}", "0.68", turn.ay / G)


def measured(names):
    sets = measured_sets("rider")
    for name in names:
        p = sets[name]
        u = upright_speeds(p)
        _, (Vbt, phibt), (V, phi), _ = region(name, p, u)
        got = (u["v_w"], u["v_c"], u["V_a3"], Vbt, phibt * DEG, phi * DEG, V)
        for col, pv, iv in zip(MEASURED_COLUMNS, MEASURED[name].split(), got):
            value("Table S measured-sets", f"{name} {col}", pv, iv)


# ---- exact statements --------------------------------------------------------------------------

TURNS = [(3.0, 0.10), (4.0, 0.15), (4.54, 0.21), (5.5, 0.05), (6.5, 0.30), (8.0, 0.40)]


def identities():
    b = Bike(BENCH)
    res = {"free": 0.0, "constant": 0.0, "pencil": 0.0}
    for V, phi in TURNS:
        f = fold_identity(b, V, phi)
        for plant in ("free", "constant"):
            res[plant] = max(res[plant], rel(*f[plant]))
            res["pencil"] = max(
                res["pencil"],
                rel(
                    pencil_coefficients(*pencil_matrices(b, V, phi, plant)),
                    b.quartic(V, phi, plant),
                ),
            )
    identity(
        "Theorem 1 (eq. fold)",
        "a4 against mu det d(T,H)/d(lean,steer), free rolling",
        res["free"],
        1e-9,
    )
    identity(
        "Sect. 6.4 Capsize rule at constant speed",
        "a4 against mu_S det d(T,S)/d(lean,steer)",
        res["constant"],
        1e-9,
    )
    identity(
        "Eq. (pencil)",
        "quartic of the 2x2 pencil against the Jacobian, both plants",
        res["pencil"],
        1e-9,
    )
    err = 0.0
    for V in (3.0, 4.5, 7.0):
        z = np.array([0.0, 0.0, 0.0, 0.0, -V / b.rR])
        err = max(
            err,
            rel(
                np.real(np.poly(b.jacobian(z, 0.0, "constant"))),
                np.r_[1.0, upright_coefficients(BENCH, V)],
            ),
        )
    identity("Eq. (K)", "upright nonlinear Jacobian against the linear model", err, 1e-9)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(10):
        z = np.r_[rng.uniform(-0.4, 0.4, 2), rng.uniform(-1, 1, 2), -rng.uniform(5, 30)]
        worst = max(worst, abs(power_residual(b, z, rng.uniform(-2, 2))))
    identity("Eq. (power)", "dE/dt - T ddelta at random states (W)", worst, 1e-10)


def _parallel(q):
    return (q @ q) * np.eye(3) - np.outer(q, q)


def _add_point(p, body, dm, x, z):
    """Add a point mass dm at (x, 0, z) to a frame (dm < 0 removes it), parallel-axis theorem."""
    q = dict(p)
    m, Q, P = p["m" + body], np.array([p["x" + body], 0, p["z" + body]]), np.array([x, 0, z])
    I0 = np.array(
        [
            [p[f"I{body}xx"], 0, p[f"I{body}xz"]],
            [0, p[f"I{body}yy"], 0],
            [p[f"I{body}xz"], 0, p[f"I{body}zz"]],
        ]
    )
    mn = m + dm
    Qn = (m * Q + dm * P) / mn
    In = I0 + m * _parallel(Q) + dm * _parallel(P) - mn * _parallel(Qn)
    q.update(
        {
            "m" + body: mn,
            "x" + body: Qn[0],
            "z" + body: Qn[2],
            f"I{body}xx": In[0, 0],
            f"I{body}yy": In[1, 1],
            f"I{body}zz": In[2, 2],
            f"I{body}xz": In[0, 2],
        }
    )
    return q


def transforms(p, a=1.2):
    """Parameter changes and their scale factors (speed, torque, eigenvalue) of Sect. 12.1."""
    out = {}
    out["1 total mass"] = ({k: v * a if k[0] in "mI" else v for k, v in p.items()}, 1, a, 1)
    for wheel, frame, x, z in (("R", "B", 0.0, -p["rR"]), ("F", "H", p["w"], -p["rF"])):
        q = _add_point(p, frame, 0.2 * p["m" + wheel], x, z)
        q["m" + wheel] *= 0.8
        out[f"2 wheel mass {wheel} to frame"] = (q, 1, 1, 1)
        e = 0.2 * p[f"I{wheel}xx"]
        q = dict(
            p,
            **{
                f"I{wheel}xx": p[f"I{wheel}xx"] - e,
                f"I{frame}xx": p[f"I{frame}xx"] + e,
                f"I{frame}zz": p[f"I{frame}zz"] + e,
            },
        )
        out[f"2 wheel diametral inertia {wheel} to frame"] = (q, 1, 1, 1)
    for z in (-1.0, -0.6, -0.2):
        x = p["w"] + p["c"] + z * math.tan(p["lam"])
        dm = 0.2 * p["mH"]
        out[f"3 mass on steer axis at z = {z}"] = (
            _add_point(_add_point(p, "B", -dm, x, z), "H", dm, x, z),
            1,
            1,
            1,
        )
    lengths = ("w", "c", "rR", "rF", "xB", "zB", "xH", "zH")
    out["(a) gravity"] = (dict(p, g=a * p["g"]), a**0.5, a, a**0.5)
    out["(a) lengths"] = (
        {k: v * a if k in lengths else v * a * a if k[0] == "I" else v for k, v in p.items()},
        a**0.5,
        a,
        a**-0.5,
    )
    out["(b) lengths, g and masses"] = (
        {
            k: v * a if k in lengths or k == "g" else v / a**2 if k[0] == "m" else v
            for k, v in p.items()
        },
        a,
        1,
        1,
    )
    return out


def invariances():
    b = Bike(BENCH)
    qs = [(0.05, 0.025), (0.13, 0.075), (0.23, 0.15)]

    def spectrum(bike, q):
        S, T = bike.static(*q)
        z = np.array([q[0], q[1], 0.0, 0.0, -math.sqrt(S)])
        return (
            S,
            T,
            [
                np.sort_complex(np.linalg.eigvals(bike.jacobian(z, T, pl)))
                for pl in ("free", "constant")
            ],
        )

    base = [spectrum(b, q) for q in qs]
    for label, (p, vs, ts, es) in transforms(b.p).items():
        bb = Bike(p)
        err = 0.0
        for q, (S, T, ev) in zip(qs, base):
            S2, T2, ev2 = spectrum(bb, q)
            sf = vs * b.rR / bb.rR
            err = max(err, abs(S2 / (S * sf * sf) - 1), abs(T2 / (ts * T) - 1))
            for e1, e2 in zip(ev, ev2):
                err = max(err, rel(e2 / es, e1))
        identity(
            "Sect. 12.1 Exact invariances", f"statement {label}: turns and eigenvalues", err, 1e-9
        )
    err = 0.0
    for body in ("B", "H"):
        e = 0.2 * min(b.p[f"I{body}{k}"] for k in ("xx", "yy", "zz"))
        bb = b.with_params(**{f"I{body}{k}": b.p[f"I{body}{k}"] + e for k in ("xx", "yy", "zz")})
        for q in qs:
            err = max(err, rel(bb.static(*q), b.static(*q)))
    identity(
        "Sect. S invariances (c)",
        "isotropic frame inertia: squared spin and steer torque",
        err,
        1e-9,
    )
    t0 = tau3(b, 4.5)
    t1 = tau3(b.with_params(IBxx=1.2 * b.p["IBxx"], IHxx=1.2 * b.p["IHxx"]), 4.5)
    identity("Sect. S invariances (d)", "roll inertias: relative change of tau3", t1 / t0 - 1, 1e-6)


def parameter_rank(name, p, n_states=16):
    """Rank of d(M, f)/d(parameters) over the 26 parameters, by complex step, at random states."""
    keys = [k for k in PARAMETERS if k != "yB"]
    rng = np.random.default_rng(3)
    states = np.column_stack(
        [
            rng.uniform(0.02, 0.35, n_states),
            rng.uniform(0.01, 0.22, n_states),
            rng.uniform(-0.4, 0.5, (n_states, 2)),
            rng.uniform(-25, -8, n_states),
        ]
    )
    base = {k: p.get(k, 9.81) for k in keys}

    def observe(x):
        from whipple.model import Whipple

        m = Whipple(dict(zip(keys, x)))
        rows = []
        for s in states:
            _, M, f, _ = m.parts(s[0], s[1], s[2:])
            rows.extend([*M[np.triu_indices(3)], *f])
        return np.array(rows)

    x0 = np.array([base[k] for k in keys])
    J = complex_jacobian(observe, x0) * np.maximum(np.abs(x0), 1e-2)
    J /= np.maximum(np.max(np.abs(J), axis=1), 1e-300)[:, None]
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-8 * sv[0]))
    ROWS.append(
        dict(
            where="Sect. S invariances",
            quantity=f"{name}: rank of d(M, f)/d(parameters)",
            paper="18",
            independent=rank,
            difference=rank - 18,
            agrees=rank == 18,
            singular_values_18_19=[float(sv[17]), float(sv[18])],
        )
    )


# ---- energy ------------------------------------------------------------------------------------


def capsize_run():
    """Free-rolling capsize from the unstable turn at 4.54 m/s and 12.6 deg (Sect. 3.2)."""
    b = Bike(BENCH)
    z, T = b.trim(4.54, math.radians(12.6))
    vals, vecs = np.linalg.eig(b.jacobian(z, T))
    i = int(np.argmax(vals.real))
    H = b.energy(z) - T * z[1]
    z = z + 0.001 * np.real(vecs[:, i] / vecs[0, i])
    # Restore the first integral H = E - T delta by the spin.
    M = b.model.parts(z[0], z[1], np.zeros(3))[1]
    qa, qb = 0.5 * M[2, 2], M[2, :2] @ z[2:4]
    qc = 0.5 * z[2:4] @ M[:2, :2] @ z[2:4] + b.model.potential(z[0], z[1]) - T * z[1] - H
    z[4] = (-qb - math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)

    def at45(t, y):
        return math.pi / 4 - abs(y[0])

    at45.terminal = True
    sol = solve_ivp(
        lambda t, y: b.field(y, T),
        (0, 2000),
        z,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.15,
        events=at45,
    )
    end = sol.y[:, -1]
    m = b.model
    qd, Me, _, _ = m.parts(end[0], end[1], end[2:])
    jv = m._geometry(end[0], end[1])[2]
    v_cm = np.einsum("b,bki,i->k", m.mass, jv, qd) / m.mass.sum()
    v_rear = np.hypot(
        qd[0] + b.rR * math.sin(end[0]) * qd[3], qd[1] - b.rR * math.cos(end[0]) * qd[4]
    )
    drop = m.potential(z[0], z[1]) - m.potential(end[0], end[1])
    dh = drop / (m.mass.sum() * G)

    def k_forward(y):
        Mm = m.parts(y[0], y[1], np.zeros(3))[1]
        return (Mm[2] @ y[2:5]) ** 2 / (2 * Mm[2, 2])

    w = "Sect. 3.2 Energy in disturbed motions"
    value(w, "potential energy drop from 12.6 to 45 deg (J)", "214.7", drop)
    value(
        w,
        "share of the drop taken by K_f (%)",
        "99.4",
        100 * (k_forward(end) - k_forward(z)) / drop,
    )
    value(
        w,
        "speed estimate sqrt(V0^2 + 2 g dh) (m/s)",
        "5.02",
        math.sqrt((z[4] * b.rR) ** 2 + 2 * G * dh),
    )
    value(w, "rear-contact speed at 45 deg (m/s)", "6.03", v_rear)
    value(w, "mass-centre speed at 45 deg (m/s)", "4.83", np.linalg.norm(v_cm))
    identity(w, "drift of H = E - T delta over the run (J)", b.energy(end) - T * end[1] - H, 1e-8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="five measured sets, coarser grid")
    args = ap.parse_args()
    start = time.time()

    def stage(name, f, *a):
        t = time.time()
        out = f(*a)
        print(f"{name}: {time.time() - t:.0f} s", flush=True)
        return out

    u, kstar = stage("upright and spring", upright_and_spring)
    stage("identities", identities)
    stage("invariances", invariances)
    for name, p in (
        ("benchmark", BENCH),
        ("Browser+Jason", measured_sets("rider")["Browser+Jason"]),
    ):
        stage(f"parameter rank {name}", parameter_rank, name, p)
    stage("regions", regions, u, kstar)
    stage("critical trails", critical_trails)
    stage("countersteer grid", countersteer, 9 if args.quick else 25)
    stage("capsize run", capsize_run)
    stage("measured sets", measured, QUICK_SETS if args.quick else list(MEASURED))
    print(f"{'quantity':62s} {'paper':>9s} {'independent':>14s} {'difference':>11s}")
    for r in ROWS:
        flag = "" if r["agrees"] else "  <-- differs"
        print(
            f"{r['quantity'][:62]:62s} {r['paper']:>9s} {r['independent']:14.6g} "
            f"{float('nan') if r['difference'] is None else r['difference']:11.2e}{flag}"
        )
    n_bad = sum(not r["agrees"] for r in ROWS)
    print(f"{len(ROWS)} rows, {n_bad} differ, {time.time() - start:.0f} s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"quick": args.quick, "rows": ROWS}, indent=1) + "\n")


if __name__ == "__main__":
    main()
