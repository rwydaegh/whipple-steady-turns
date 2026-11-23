"""Parameter sets and analysis steps shared by the scripts on the measured sets (Sect. 11,
"Measured bicycles").
"""

import gzip
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from whipple import capsize_rule  # noqa: E402
from whipple import stability as st  # noqa: E402
from whipple.bicycles import BICYCLES, measured_sets  # noqa: E402
from whipple.linear import Linear  # noqa: E402
from whipple.quartic import bracket  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "measured"
RESULTS = DATA / "results"
G = 9.81

# Bicycle A with an invented rigid 75 kg rider (torso 10 deg from the vertical) in the rear frame.
A_RIDER = dict(
    BICYCLES["A"],
    xB=0.3323805724636031,
    zB=-1.0227853258556654,
    mB=84.18,
    IBxx=14.557159213220098,
    IByy=14.18911559953214,
    IBzz=2.237861374031043,
    IBxz=-0.1646269237726708,
)


def all_sets():
    """name -> (params, group, partner): the measured sets, the benchmark, A and A with rider.

    ``partner`` is the riderless set of the same bicycle (None if there is none). Silver is left
    out: its region is open and its raw data are incomplete.
    """
    sets = {}
    for k, p in measured_sets("riderless").items():
        if k != "Silver":
            sets[k] = (p, "riderless", None)
    for k, p in measured_sets("rider").items():
        bike = k.split("+")[0].replace("DelftBrowser", "")
        sets[k] = (p, "rider", bike if bike in sets else None)
    sets["benchmark"] = (dict(BICYCLES["benchmark"]), "rider (benchmark, hypothetical)", None)
    sets["A_rider(invented)"] = (dict(A_RIDER), "rider (invented 75 kg)", "A")
    sets["A"] = (dict(BICYCLES["A"]), "riderless", None)
    return sets


# ---- analysis steps ------------------------------------------------------------------------------
def upright(p):
    """Stable speed intervals on a 0.01 m/s grid, v_c, v_w and V_a3 of the upright bicycle."""
    lin = Linear(p)
    vs = np.arange(0.2, 30.0, 0.01)
    stable = np.array([lin.eigenvalues(v).real.max() for v in vs]) < 0
    r = {"stable_intervals_grid": []}
    on = None
    for i in np.flatnonzero(np.diff(stable.astype(int)) != 0):
        if stable[i + 1]:
            on = vs[i]
        elif on is not None:
            r["stable_intervals_grid"].append([float(on), float(vs[i])])
            on = None
    if on is not None:
        r["stable_intervals_grid"].append([float(on), 30.0])
    try:
        r["v_c"] = lin.capsize_speed()
    except ValueError:
        r["v_c"] = math.nan
    vw = math.nan
    if r["stable_intervals_grid"]:
        v0 = r["stable_intervals_grid"][0][0]
        try:
            vw = lin.weave_speed((v0 - 0.02, v0 + 0.02))
        except ValueError:
            vw = math.nan
    r["v_w"] = vw
    r["V_a3"] = v_a3(lin)
    return r


def v_a3(lin):
    """Speed where the upright coefficient a3 is zero (eq. Va3), as Quartic.v_a3; nan if none."""
    q = -lin.g * bracket(lin.C1, lin.K0) / bracket(lin.C1, lin.K2)
    return math.sqrt(q) if q > 0 else math.nan


def torque_per_g(bike, p, V, phi=0.08726646259971647):  # 5 deg
    """Steady steer torque per g of lateral acceleration (N m) at speed V and lean phi.

    Returns (value, note). The steer of the turn comes from Newton's method; where it fails,
    from bracketing (note "by bracketing").
    """
    g0 = phi * Linear(p).steer_ratio(V)
    try:
        d = st.delta_for_speed(bike, phi, V, g0)
        note = None
    except (RuntimeError, ValueError):

        def res(d):
            t = bike.turn(phi, d)
            return (t.V if t else 0.0) - V

        d = brentq(res, 0.5 * g0, 2.0 * g0, xtol=1e-15)
        note = "by bracketing"
    t = bike.turn(phi, d)
    return t.T / (t.ay / G), note


def _valid_mask(bike, curve, kind, plant):
    """True where the eigenvalues other than the crossing ones have negative real parts."""
    ok = []
    for x in curve:
        a1, a2, a3, a4 = st.characteristic(bike, *x, plant)
        if kind == "hopf":
            ok.append(a1 > 0 and a3 > 0 and a4 > 0)
        else:
            ok.append(a1 > 0 and a3 > 0 and a1 * a2 - a3 > 0)
    return np.array(ok)


def region(bike, plant, vw):
    """Region of stable turns of ``plant`` ("free" or "constant"): tip, extremes and curves."""
    t = st.tongue(bike, plant=plant, phi_max=0.8, ds_max=4e-3, weave_bracket=(vw - 0.05, vw + 0.05))
    tests = {
        "hopf": lambda x: st.hurwitz(st.characteristic(bike, *x, plant))[0],
        "fold": lambda x: st.characteristic(bike, *x, plant)[3],
    }
    bt = st.trimmed_turn(bike, *t.bt, plant)
    a_bt = st.characteristic(bike, *t.bt, plant)
    res = {"BT_lean_deg": math.degrees(bt.phi), "BT_V": bt.V, "BT_T": bt.T, "BT_a2": float(a_bt[1])}
    best, best_ay, vmin, frac = None, None, math.inf, {}
    for name, curve in (("hopf", t.hopf[:-1]), ("fold", t.fold[:-1])):
        m = _valid_mask(bike, curve, name, plant)
        frac[name] = float(m.mean())
        turns = [st.trimmed_turn(bike, *x, plant) for x in curve]
        for q, key in ((lambda u: u.phi, "lean"), (lambda u: u.ay, "ay")):
            vals = np.array([q(u) if ok else -np.inf for u, ok in zip(turns, m)])
            if not np.isfinite(vals).any():
                continue
            i = int(np.argmax(vals))
            x = curve[i]
            if 0 < i < len(curve) - 1 and m[i - 1] and m[i + 1]:
                try:
                    x = st.boundary_extremum(bike, tests[name], q, x, plant)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    pass
            u = st.trimmed_turn(bike, *x, plant)
            cand = (q(u), name, u)
            if key == "lean" and (best is None or cand[0] > best[0]):
                best = cand
            if key == "ay" and (best_ay is None or cand[0] > best_ay[0]):
                best_ay = cand
        vv = [u.V for u, ok in zip(turns, m) if ok]
        if vv:
            vmin = min(vmin, min(vv))
    res.update(
        valid_fraction=frac,
        tip_type="BT" if a_bt[1] > 0 else "zero-Hopf (a2<0 at BT)",
        lowest_stable_V=vmin if vmin < math.inf else math.nan,
    )
    if best is not None:
        u = best[2]
        res.update(
            max_lean_deg=math.degrees(u.phi),
            V_at_max_lean=u.V,
            max_lean_curve=best[1],
            ay_at_max_lean_g=u.ay / G,
            T_at_max_lean=u.T,
        )
    if best_ay is not None:
        res.update(max_ay_g=best_ay[2].ay / G, V_at_max_ay=best_ay[2].V)
    res["curves"] = {"hopf": t.hopf.tolist(), "fold": t.fold.tolist()}
    res["curves_V"] = {
        k: [st.trimmed_turn(bike, *x, plant).V for x in c]
        for k, c in (("hopf", t.hopf), ("fold", t.fold))
    }
    return res


def grid_scan(bike, plant="free"):
    """Turns on a grid of 1 to 45 deg of lean (1 deg steps) by 40 steer angles.

    Each row: lean (deg), steer (rad), V (m/s), a_y/g, T (N m), stable (bool).
    """
    pts = []
    for phi in np.radians(np.arange(1.0, 45.1, 1.0)):
        for d in np.geomspace(0.002, 0.6, 40):
            u = bike.turn(phi, d)
            if u is None or not (0.5 < u.V < 15):
                continue
            a = st.characteristic(bike, phi, d, plant)
            stable = bool(np.all(np.roots(np.r_[1.0, a]).real < 0))
            pts.append([math.degrees(phi), float(d), u.V, u.ay / G, u.T, stable])
    return pts


# Stored rule keys: plant, key prefix, (order of largest_lean_rule, suffix).
RULE_FORMS = {
    "constant": ((1, "cubic"), (2, "quintic")),
    "free": ((1, "first_order"), (2, "second_order")),
}


def _plain(v):
    return float(v) if isinstance(v, (float, np.floating)) else v


def rule_block(p):
    """Capsize rules: top of the fold curve from each rule, and the closed form at v_c."""
    out = {}
    for plant in ("constant", "free"):
        for order, suffix in RULE_FORMS[plant]:
            try:
                r = capsize_rule.largest_lean_rule(p, plant, order)
                out[f"{plant}_{suffix}"] = {
                    k: _plain(v) for k, v in r.items() if k != "speed_range"
                }
            except (RuntimeError, ValueError, ZeroDivisionError) as e:
                out[f"{plant}_{suffix}"] = {"error": repr(e)}
        try:
            c = capsize_rule.largest_lean_closed(p, plant)
            out[f"{plant}_closed"] = {k: float(v) for k, v in c.items()}
        except (RuntimeError, ValueError, ZeroDivisionError) as e:
            out[f"{plant}_closed"] = {"error": repr(e)}
    try:
        out["speed_range"] = [
            z if isinstance(z, str) else float(z) for z in capsize_rule.speed_range(p)
        ]
    except (RuntimeError, ValueError) as e:
        out["speed_range"] = repr(e)
    return out


# ---- inertia tensors -----------------------------------------------------------------------------
def tensor(p, body):
    """Inertia tensor of body "B" (rear frame) or "H" (front frame) about its mass centre."""
    xx, yy, zz, xz = (p[f"I{body}{k}"] for k in ("xx", "yy", "zz", "xz"))
    return np.array([[xx, 0.0, xz], [0.0, yy, 0.0], [xz, 0.0, zz]])


def excess(inertia):
    """Triangle excess e = I_max - I_mid - I_min of the principal moments, and the moments.

    A rigid body has e <= 0.
    """
    lam = np.sort(np.linalg.eigvalsh(inertia))
    return float(lam[2] - lam[1] - lam[0]), lam


# ---- stored results ------------------------------------------------------------------------------
def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def load_result(name, folder=RESULTS):
    return read_json(Path(folder) / f"{name}.json.gz")


def all_results(folder=RESULTS):
    out = {}
    for f in sorted(Path(folder).glob("*.json.gz")):
        d = read_json(f)
        out[d["name"]] = d
    return out


def write_json_gz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f, default=float)
