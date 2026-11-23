"""Exact region and rule predictions of one random design (supplement, "Random designs")."""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from sampler import load  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from whipple import capsize_rule as cr  # noqa: E402
from whipple import stability as st  # noqa: E402
from whipple.lean_law import LeanLaw  # noqa: E402
from whipple.linear import Linear  # noqa: E402
from whipple.model import Whipple  # noqa: E402
from whipple.quartic import Quartic, hurwitz_d3  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "random_designs" / "out"
DEG = 180 / math.pi
OPEN_LEAN = 45.0  # deg; a stable component that reaches this lean is open
PLANTS = ("free", "constant")
ORDERS = {"first_order": 1, "second_order": 2}


# ---- upright ------------------------------------------------------------------------------------
def stable_band(lin, vmax=40.0, n=700):
    """First speed interval (v_lo, v_hi) with all upright eigenvalues damped (v_hi inf if open)."""

    def growth(v):
        return lin.eigenvalues(v).real.max()

    vs = np.geomspace(0.3, vmax, n)
    s = np.array([growth(v) for v in vs]) < 0
    if not s.any():
        return math.nan, math.nan
    i = int(np.argmax(s))
    lo = brentq(growth, vs[i - 1], vs[i], xtol=1e-10) if i > 0 else 0.0
    rest = np.flatnonzero(~s[i:])
    if not len(rest):
        return lo, math.inf
    j = i + int(rest[0])
    return lo, brentq(growth, vs[j - 1], vs[j], xtol=1e-10)


def lin_consts(p):
    K0 = Linear(p).K0
    mT = p["mR"] + p["mB"] + p["mH"] + p["mF"]
    zT = (-p["rR"] * p["mR"] + p["zB"] * p["mB"] + p["zH"] * p["mH"] - p["rF"] * p["mF"]) / mT
    SF = p["IFyy"] / p["rF"]
    return dict(mTzT=-mT * zT, SA=-K0[0, 1], SF=SF, ST=SF + p["IRyy"] / p["rR"],
                sl=math.sin(p["lam"]))  # fmt: skip


# ---- turns on a grid ----------------------------------------------------------------------------
def solve_delta(bike, phi, V, guess, tol=1e-13, maxiter=30):
    """Steer of the turn at lean phi and contact speed V (Newton, complex-step slope)."""
    s2t = (V / bike.p["rR"]) ** 2
    d = guess
    for _ in range(maxiter):
        s2 = bike.turn_spin_torque(phi, d + 1e-30j)[0]
        f, df = s2.real - s2t, s2.imag / 1e-30
        if not np.isfinite(f) or df == 0:
            return math.nan
        step = max(-0.05, min(0.05, f / df))
        d -= step
        if abs(step) < tol:
            return d
    return math.nan


def stable_of(a):
    return bool(np.all(np.roots(np.r_[1.0, a]).real < 0))


def column(bike, lin, V, phis, plants=PLANTS):
    """Steer, quartic and stability along lean at speed V (continuation from upright)."""
    n = len(phis)
    delta = np.full(n, np.nan)
    A = {pl: np.full((n, 4), np.nan) for pl in plants}
    S = {pl: np.zeros(n, bool) for pl in plants}
    prev = []
    for j, phi in enumerate(phis):
        if len(prev) >= 2:
            (p1, d1), (p2, d2) = prev[-2], prev[-1]
            guess = d2 + (d2 - d1) * (phi - p2) / (p2 - p1)
        else:
            guess = prev[-1][1] * phi / prev[-1][0] if prev else lin.steer_ratio(V) * phi
        d = solve_delta(bike, phi, V, guess)
        if not d == d or abs(d) > 1.4:
            break
        t = bike.turn(phi, d)
        if t is None or abs(t.V - V) > 1e-7:
            break
        prev.append((phi, d))
        delta[j] = d
        for pl in plants:
            try:
                A[pl][j] = st.characteristic(bike, phi, d, pl)
                S[pl][j] = stable_of(A[pl][j])
            except Exception:  # noqa: BLE001
                continue
    return delta, A, S


def grid_eval(bike, lin, Vs, phis):
    cols = [column(bike, lin, V, phis) for V in Vs]
    delta = np.array([c[0] for c in cols])
    A = {pl: np.array([c[1][pl] for c in cols]) for pl in PLANTS}
    S = {pl: np.array([c[2][pl] for c in cols]) for pl in PLANTS}
    return delta, A, S


def component(stable, Vs, v_w, v_hi):
    """Stable component connected to the upright stable band."""
    lab, _ = ndimage.label(stable)
    seed = np.flatnonzero((Vs > v_w) & (Vs < v_hi))
    ids = set(lab[seed, 0].tolist()) - {0}
    return np.isin(lab, list(ids))


# ---- top and tip ---------------------------------------------------------------------------------
def tests(bike, plant):
    a = lambda x: st.characteristic(bike, *x, plant)  # noqa: E731
    return {"fold": lambda x: a(x)[3], "hopf": lambda x: hurwitz_d3(a(x)),
            "corner": lambda x: np.array([a(x)[3], hurwitz_d3(a(x))])}  # fmt: skip


def refine_top(bike, plant, x0, kind):
    """Largest-lean point near x0 = (phi, delta) on curve ``kind`` (fold, hopf or corner)."""
    T = tests(bike, plant)
    if kind == "corner":
        return st.newton_system(T["corner"], np.asarray(x0, float), h=1e-6, tol=1e-10)
    return st.boundary_extremum(bike, T[kind], lambda t: t.phi, x0, plant)


def check_top(bike, plant, x):
    """The turn 0.15 deg below the point, at the same speed, is stable."""
    t = bike.turn(*x)
    if t is None:
        return False
    d = solve_delta(bike, x[0] - math.radians(0.15), t.V, x[1] * (x[0] - 0.0026) / x[0])
    return d == d and stable_of(st.characteristic(bike, x[0] - math.radians(0.15), d, plant))


def find_top(bike, plant, Vs, phis, delta, A, comp):
    """Largest stable lean: grid top, then Newton on the fold, the Hopf curve or their corner."""
    jt = int(np.flatnonzero(comp.any(axis=0)).max())
    out = dict(grid_top_deg=float(phis[jt] * DEG))
    rows = np.flatnonzero(comp[:, jt])
    i = int(rows[len(rows) // 2])
    out["grid_top_V"] = float(Vs[i])
    if jt + 1 >= len(phis):
        out["top_method"] = "edge"
        return out
    a_up = A[plant][i, jt + 1]
    dphi = phis[1] - phis[0]
    first = "fold" if (a_up[3] == a_up[3] and a_up[3] < 0) else "hopf"
    phi0 = phis[jt] + 0.5 * dphi
    x0 = np.array([phi0, delta[i, jt] * phi0 / phis[jt]])
    cands = []
    for kind in (first, "hopf" if first == "fold" else "fold", "corner"):
        try:
            x = refine_top(bike, plant, x0, kind)
        except Exception:  # noqa: BLE001
            continue
        t = bike.turn(*x)
        if t is None or abs(x[0] - phis[jt]) > 2.0 * dphi or not (Vs[0] <= t.V <= Vs[-1]):
            continue
        if check_top(bike, plant, x):
            cands.append((x[0], kind, x, t))
    if cands:
        ok = [c for c in cands if c[0] >= phis[jt] - 1e-9] or cands
        c = min(ok, key=lambda c: c[0])
        out.update(top_deg=float(c[0] * DEG), top_V=c[3].V, top_kind=c[1],
                   top_x=[float(c[2][0]), float(c[2][1])], top_method="newton")  # fmt: skip
    else:
        out["top_method"] = "grid"
    return out


def verify_tip(bike, plant, x):
    """A stable turn exists just above the tip speed near the tip lean."""
    t = bike.turn(*x)
    if t is None:
        return False
    for dv, lo, hi, n in ((0.03, -2.0, 2.0, 81), (0.1, -1.0, 3.0, 401), (0.25, -1.0, 4.0, 251)):
        V, d = t.V + dv, None
        for ph in np.radians(np.linspace(max(0.1, x[0] * DEG + lo), x[0] * DEG + hi, n)):
            d = solve_delta(bike, ph, V, x[1] * ph / x[0] if d is None else d)
            if not d == d:
                d = None
                continue
            if stable_of(st.characteristic(bike, ph, d, plant)):
                return True
    return False


def find_tip(bike, plant, Vs, phis, delta, comp):
    """Tip of the stable component: a Bogdanov-Takens point with a2 > 0, else a zero-Hopf point."""
    i0 = int(np.flatnonzero(comp.any(axis=1)).min())
    cs = np.flatnonzero(comp[i0])
    j0 = int(cs[len(cs) // 2])
    out = dict(grid_tip_V=float(Vs[i0]), grid_tip_deg=float(phis[j0] * DEG))
    x0 = np.array([phis[j0], delta[i0, j0]])
    dV = Vs[min(i0 + 1, len(Vs) - 1)] - Vs[i0]
    lo = Vs[max(i0 - 1, 0)] - 0.35 * Vs[i0]
    try:
        x = st.bogdanov_takens(bike, x0, plant)
        t = bike.turn(*x)
        a = st.characteristic(bike, *x, plant)
        out.update(bt_V=t.V, bt_deg=x[0] * DEG, bt_a2=float(a[1]))
        if a[1] > 0 and lo < t.V < Vs[i0] + dV and verify_tip(bike, plant, x):
            out.update(
                tip_V=t.V, tip_deg=x[0] * DEG, tip_kind="BT", tip_x=[float(x[0]), float(x[1])]
            )
            return out
    except Exception:  # noqa: BLE001
        pass

    def F(x):
        a = st.characteristic(bike, *x, plant)
        return np.array([a[3], a[2] - a[0] * a[1]])

    try:
        x = st.newton_system(F, x0, h=1e-6, tol=1e-10)
        t = bike.turn(*x)
        if lo < t.V < Vs[i0] + dV and verify_tip(bike, plant, x):
            out.update(
                tip_V=t.V, tip_deg=x[0] * DEG, tip_kind="zH", tip_x=[float(x[0]), float(x[1])]
            )
            return out
    except Exception:  # noqa: BLE001
        pass
    out["tip_kind"] = "grid"
    return out


def second_pass(bike, lin, plant, q):
    """Fine scan for a top that the grid Newton step missed (41 speeds, 0.2 deg steps, then
    bisection to 0.005 deg and Newton)."""
    gt, gV = q["grid_top_deg"], q["grid_top_V"]
    Vs = np.linspace(max(0.5, gV - 1.5), min(gV + 1.5, q["comp_max_V"] + 0.3), 41)
    phis = np.radians(np.arange(0.2, gt + 4.0, 0.2))
    best = None
    for V in Vs:
        ds, _, S = column(bike, lin, V, phis, (plant,))
        stab = S[plant]
        idx = np.flatnonzero(stab & (phis <= math.radians(gt - 0.5)))
        if not len(idx):
            continue
        j = idx.max()
        while j + 1 < len(phis) and stab[j + 1]:
            j += 1
        if best is None or phis[j] > best[0]:
            best = (phis[j], V, ds[j], j < len(phis) - 1)
    if best is None:
        return {}
    lo, V, dlo, closed = best
    hi = lo + math.radians(0.2)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        dm = solve_delta(bike, mid, V, dlo * mid / lo)
        if dm == dm and stable_of(st.characteristic(bike, mid, dm, plant)):
            lo, dlo = mid, dm
        else:
            hi = mid
    if not closed:
        return {}
    out = dict(top_deg=lo * DEG, top_V=V, top_kind="fine grid", top_method="second pass")
    for kind in ("fold", "hopf", "corner"):
        try:
            x = refine_top(bike, plant, np.array([lo, dlo]), kind)
            if abs(x[0] - lo) * DEG < 0.3 and check_top(bike, plant, x):
                out.update(top_deg=x[0] * DEG, top_V=bike.turn(*x).V, top_kind=kind)
                break
        except Exception:  # noqa: BLE001
            continue
    return out


# ---- weave side ---------------------------------------------------------------------------------
def first_root(f, x, rising):
    s = f >= 0 if rising else f <= 0
    k = np.flatnonzero(~s[:-1] & s[1:])
    if not len(k):
        return math.nan
    i = k[0]
    return x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i])


def leans_per_speed(qu, Vs, phis, A, plant):
    """Per grid speed: exact Hopf and fold leans, the hybrid Hopf lean (exact a4, second-order
    a1, a2, a3) and the second-order Hopf lean (rad), from splines of the exact a_i in lean."""
    res = {k: np.full(len(Vs), np.nan) for k in ("hopf", "fold", "hopf_hyb", "hopf_o2")}
    for i, V in enumerate(Vs):
        a = A[plant][i]
        ok = np.isfinite(a[:, 0])
        m = int(np.argmin(ok)) if not ok.all() else len(ok)
        if m < 4:
            continue
        a0, a2 = qu.a0(V), qu.a2(V, plant)
        ph = np.r_[0.0, phis[:m]]
        x = np.linspace(0, ph[-1], 1500)
        ax = CubicSpline(ph, np.vstack([a0, a[:m]]), axis=0)(x).T
        res["hopf"][i] = first_root(hurwitz_d3(ax), x, True)
        res["fold"][i] = first_root(ax[3], x, False)
        o2 = a0[:, None] + a2[:, None] * x[None, :] ** 2
        res["hopf_o2"][i] = first_root(hurwitz_d3(o2), x, True)
        o2[3] = ax[3]
        res["hopf_hyb"][i] = first_root(hurwitz_d3(o2), x, True)
    return res


def weave_side(qu, Vs, phis, A, plant, r, v_w, v_c):
    L = leans_per_speed(qu, Vs, phis, A, plant)
    tipV = r.get("tip_V", r.get("grid_tip_V"))
    sel = (Vs > tipV) & (Vs < v_w) & np.isfinite(L["hopf"]) & (L["hopf"] < L["fold"])
    for k in ("hopf_hyb", "hopf_o2"):
        e = (L[k] - L["hopf"])[sel] * DEG
        e = e[np.isfinite(e)]
        r["err_" + k] = float(e[np.argmax(np.abs(e))]) if len(e) else math.nan
    okf = np.isfinite(L["fold"]) & (Vs < (v_c if v_c == v_c else Vs[-1]))
    if okf.sum() > 4 and tipV == tipV:
        sp = CubicSpline(Vs[okf], L["fold"][okf])

        def h(V):  # hybrid tip: exact fold lean with the second-order a3 = 0
            return qu.a0(V)[2] + qu.a2(V, plant)[2] * sp(V) ** 2

        vv = np.linspace(Vs[okf][0], Vs[okf][-1], 300)
        hv = np.array([h(v) for v in vv])
        kk = np.flatnonzero(hv[:-1] * hv[1:] < 0)
        if len(kk):
            k0 = kk[np.argmin(np.abs(vv[kk] - tipV))]
            Vh = brentq(h, vv[k0], vv[k0 + 1])
            r.update(hyb_tip_V=float(Vh), hyb_tip_deg=float(sp(Vh) * DEG))


# ---- predictions --------------------------------------------------------------------------------
def weave_margin(p, law, phi, V, plant):
    """Real part of the weave eigenvalue at lean phi and speed V from the small-lean expansion
    lambda_w + Lambda_w phi^2. Negative: the weave is damped at the top of the rule's fold curve."""
    if plant == "constant":
        t = law.terms(V)[0]
        Lam = t["inertia"] + t["gyroscopic"] + t["gravity"] + t["centrifugal"]
    else:
        t = law.terms(V + cr.coefficients(p, V)["e2"] * phi**2)[0]
        Lam = t["Lambda"]
    if abs(t["lam"].imag) < 1e-9:
        return math.nan
    return float((t["lam"] + Lam * phi**2).real)


def predictions(p, lin, qu, v_c):
    """Capsize rules (both orders, both plants), their conditions, the closed form, the type
    test D_free, V_a3 and the second-order tip."""
    s = cr.series(p)
    k = lin_consts(p)
    out = dict(SA=k["SA"], V_a3=float(qu.v_a3()))
    if v_c == v_c:
        c = s.with_derivatives(v_c)
        out.update(tau3_vc=c["tau3"], dtau1_vc=c["dtau1"], h2_vc=c["h2"],
                   D_free=3 * c["tau3"] - 2 * c["h2"] * c["dtau1"])  # fmt: skip
    for plant in PLANTS:
        r = {}
        for form, order in ORDERS.items():
            try:
                q = cr.largest_lean_rule(p, plant, order)
                r.update({form: q["phi_deg"], form + "_V": q["V"],
                          form + "_valid": bool(q.get("valid", False))})  # fmt: skip
                if order == 1:
                    r["direction"] = q.get("direction")
                    if q["phi"] == q["phi"]:
                        r["weave_margin"] = weave_margin(p, qu.law, q["phi"], q["V"], plant)
            except Exception as e:  # noqa: BLE001
                r[form + "_err"] = repr(e)[:120]
        try:
            c1 = cr.largest_lean_closed(p, plant)
            r["closed"], r["closed_V"] = c1["phi_deg"], c1["V"]
        except Exception as e:  # noqa: BLE001
            r["closed_err"] = repr(e)[:120]
        try:
            b = qu.bt(plant)
            if b is not None:
                r.update(o2_tip_V=float(b[0]), o2_tip_deg=float(b[1] * DEG),
                         o2_a2=float(qu.a(b[0], b[1], plant)[1]))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            r["o2_err"] = repr(e)[:120]
        out[plant] = r
    return out


def classify_pred(pr, v_w, v_c, v_hi, plant):
    """Type from upright data: sign of tau3(v_c) (constant speed) or D_free (free rolling); tip
    type from the sign of the second-order a2 at the second-order tip."""
    if pr["SA"] <= 0 or not v_w == v_w:
        return "no_band"
    if not v_c == v_c:
        return "no_vc" if v_hi == math.inf else "no_band"
    if abs(v_hi - v_c) > 1e-3 * v_c:
        return "no_band"
    if (pr["tau3_vc"] if plant == "constant" else pr["D_free"]) < 0:
        return "open"
    a2 = pr[plant].get("o2_a2", math.nan)
    return "closed_zH" if a2 == a2 and a2 < 0 else "closed_BT"


# ---- one design ---------------------------------------------------------------------------------
def speed_grid(v_w, v_c, nlo=22, nmid=34, nhi=14):
    lo = max(0.3, 0.45 * v_w)
    if v_c == v_c:
        top, mid_hi = min(max(1.45 * v_c, v_c + 2.0), 30.0), v_c
    else:
        top = min(max(3.0 * v_w, v_w + 6.0), 30.0)
        mid_hi = min(2.0 * v_w, top - 1)
    return np.r_[
        np.linspace(lo, v_w, nlo, endpoint=False),
        np.linspace(v_w, mid_hi, nmid, endpoint=False),
        np.linspace(mid_hi, top, nhi),
    ]


def analyse(p, phi_step_deg=0.75, phi_max_deg=60.0):
    """Exact type, tip, largest stable lean and weave side of both plants, and the predictions."""
    p = {k: float(v) for k, v in p.items()}
    bike = Whipple(p)
    lin = Linear(bike.p)
    res = {"errors": []}
    v_c = cr.series(p).v_c
    v_w, v_hi = stable_band(lin)
    res.update(v_w=v_w, v_c=v_c, v_hi=v_hi)
    qu = Quartic(p, LeanLaw(p))
    try:
        res["pred"] = predictions(p, lin, qu, v_c)
        for pl in PLANTS:
            res["pred"][pl]["type"] = classify_pred(res["pred"], v_w, v_c, v_hi, pl)
    except Exception as e:  # noqa: BLE001
        res["errors"].append("pred: " + repr(e)[:200])
    if not v_w == v_w:
        res.update({pl: {"type": "no_band"} for pl in PLANTS})
        cr._CACHE.clear()
        return res
    Vs = speed_grid(v_w, v_c)
    phis = np.radians(np.arange(phi_step_deg, phi_max_deg + 1e-9, phi_step_deg))
    delta, A, S = grid_eval(bike, lin, Vs, phis)
    for pl in PLANTS:
        r = {}
        try:
            comp = component(S[pl], Vs, v_w, v_hi if v_hi == v_hi else math.inf)
            if not comp.any():
                res[pl] = {"type": "no_region"}
                continue
            top_deg = phis[np.flatnonzero(comp.any(axis=0)).max()] * DEG
            r["comp_max_V"] = float(Vs[np.flatnonzero(comp.any(axis=1)).max()])
            r.update(find_tip(bike, pl, Vs, phis, delta, comp))
            if not v_c == v_c:
                r.update(type="no_vc", grid_top_deg=float(top_deg))
            elif top_deg >= OPEN_LEAN or comp[-1].any():
                r.update(type="open", grid_top_deg=float(top_deg))
            else:
                r.update(find_top(bike, pl, Vs, phis, delta, A, comp))
                r["type"] = "closed_zH" if r.get("tip_kind") == "zH" else "closed_BT"
                # in a thin region the tip can be the largest stable lean
                if r.get("tip_kind") in ("BT", "zH") and r.get("tip_deg", -1) > r.get(
                    "top_deg", -1
                ):
                    r.update(
                        top_deg=r["tip_deg"],
                        top_V=r["tip_V"],
                        top_kind="tip_" + r["tip_kind"],
                        top_x=r["tip_x"],
                        top_method="tip",
                    )
                if r.get("top_method") != "newton" and (
                        r.get("grid_top_deg", 0) > r.get("top_deg", 0) + 0.5
                        or r.get("top_kind") == "tip_BT"):  # fmt: skip
                    r.update(second_pass(bike, lin, pl, r))
            weave_side(qu, Vs, phis, A, pl, r, v_w, v_c)
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{pl}: " + repr(e)[:200])
            r.setdefault("type", "error")
        res[pl] = r
    cr._CACHE.clear()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0, help="first design index (0-3023)")
    ap.add_argument("--stop", type=int, default=None, help="index after the last design")
    ap.add_argument("--id", nargs="*", help="design ids, e.g. rand_00005")
    a = ap.parse_args()
    designs = load()
    todo = [d for d in designs if d["id"] in a.id] if a.id else designs[a.start : a.stop]
    OUT.mkdir(parents=True, exist_ok=True)
    for d in todo:
        f = OUT / f"{d['id']}.json"
        if f.exists():
            continue
        t0 = time.time()
        r = analyse(d["params"])
        out = {"id": d["id"], "kind": d["kind"], "result": r, "seconds": time.time() - t0}
        f.write_text(json.dumps(out, default=float))
        print(
            f"{d['id']}: {out['seconds']:.0f} s, free {r['free'].get('type')} "
            f"{r['free'].get('top_deg', math.nan):.3f} deg",
            flush=True,
        )


if __name__ == "__main__":
    main()
