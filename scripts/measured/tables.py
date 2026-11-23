"""Tables and stated numbers of the measured sets (tab:measured-sets, tab:measured-riderless, Sect.
11, "Measured bicycles").
"""

import json
import math

import numpy as np
from common import DATA, all_results, excess, tensor

from whipple import capsize_rule
from whipple import stability as st
from whipple.bicycles import measured_sets, measured_sources
from whipple.linear import Linear
from whipple.model import Whipple

NAN = math.nan
MASSES = ("mR", "mB", "mH", "mF")
LIMIT = 20.0  # deg: the rule statistics use the sets with an exact largest stable lean below this


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return NAN
        d = d[k]
    return d


def fmt(x, n=2, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def mass(r):
    return sum(r["params"][k] for k in MASSES)


def band(r):
    return get(r, "upright", "v_c") - get(r, "upright", "v_w")


def tip_V(r, plant="free"):
    """Lowest stable speed: the BT speed, or the zero-Hopf point where the tip is not a BT point."""
    return (
        get(r, plant, "BT_V")
        if get(r, plant, "tip_type") == "BT"
        else get(r, plant, "lowest_stable_V")
    )


def zero_hopf_point(r):
    """Zero-Hopf point (a4 = 0, a3 = a1 a2) on the stored Hopf curve: (V, lean deg)."""
    bike = Whipple(r["params"])
    curve = np.array(r["free"]["curves"]["hopf"])
    a4 = np.array([st.characteristic(bike, *x)[3] for x in curve])
    k = int(np.flatnonzero(a4 <= 0)[0])

    def F(x):
        a = st.characteristic(bike, *x)
        return np.array([a[3], a[2] - a[0] * a[1]])

    x = st.newton_system(F, curve[k])
    t = bike.turn(*x)
    return t.V, math.degrees(t.phi)


def inadmissible(p):
    """Bodies whose inertia tensor breaks the triangle inequality, with the excess."""
    return {b: e for b in "BH" if (e := excess(tensor(p, b))[0]) > 0}


def rule_errors(R, names, form):
    """Rule minus exact largest stable lean (deg), free rolling, by set."""
    return {
        n: get(R[n], "rule", f"free_{form}", "phi_deg") - get(R[n], "free", "max_lean_deg")
        for n in names
    }


def rms(v):
    v = np.asarray(list(v), float)
    return float(np.sqrt(np.mean(v**2)))


def signed_max(d):
    k = max(d, key=lambda n: abs(d[n]))
    return d[k], k


def main():
    R = all_results()
    rider = list(measured_sets("rider"))
    sources = measured_sources("rider")
    riderless = [n for n in measured_sets("riderless") if n in R]
    S = {}  # stated numbers
    lines = []

    # ---- tab:measured-sets ---------------------------------------------------------------------
    lines += [
        "## Table tab:measured-sets: measured bicycle and rider sets, free rolling\n",
        "Speeds in m/s, leans in deg, T/g in N m at 3 m/s and 5 deg of lean. "
        "Markers: + inadmissible front-frame tensor, * torso pitch assumed, "
        "§ scaled child model.\n",
        "| Set | m (kg) | v_w | v_c | V_a3 | Tip V | Tip lean | Largest lean | at V | a_y*/g "
        "| T/g |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in sorted(rider, key=lambda s: s.lower()) + ["benchmark"]:
        r = R[n]
        mark = ""
        if "H" in inadmissible(r["params"]):
            mark += "+"
        if "assumed rider configuration" in sources.get(n, {}).get("status", ""):
            mark += "*"
        if n == "Gyro+Child":
            mark += "§"
        u, f = r["upright"], r["free"]
        lines.append(
            f"| {n}{mark} | {mass(r):.1f} | {fmt(u['v_w'])} | {fmt(u['v_c'])} | {fmt(u['V_a3'])} | "
            f"{fmt(f['BT_V'])} | {fmt(f['BT_lean_deg'])} | {fmt(f['max_lean_deg'])} | "
            f"{fmt(f['V_at_max_lean'])} | {fmt(f['max_ay_g'], 3)} | {fmt(r.get('T_per_g@3'))} |"
        )

    # ---- tab:measured-riderless ----------------------------------------------------------------
    zh = zero_hopf_point(R["Browser"])
    S["riderless Browser zero-Hopf tip (V m/s, lean deg)"] = zh
    S["riderless Browser zero-Hopf tip / v_w"] = zh[0] / get(R["Browser"], "upright", "v_w")
    lines += [
        "\n## Table tab:measured-riderless: riderless measured bicycles, free rolling\n",
        "ZH: the tip is a zero-Hopf point. Silver: upright speeds only (no stored result; its raw "
        "data are incomplete and its region is open).\n",
        "| Bicycle | m (kg) | v_w | v_c | Tip V | Tip lean | Largest lean |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in sorted(riderless, key=lambda s: s.lower()):
        r = R[n]
        u = r["upright"]
        mark = "+" if inadmissible(r["params"]) else ""
        if "free" not in r:
            row = f"| {n}{mark} | {mass(r):.1f} | {fmt(u['v_w'])} | none | -- | -- | open |"
        elif get(r, "free", "tip_type") != "BT":
            row = (
                f"| {n}{mark} | {mass(r):.1f} | {fmt(u['v_w'])} | {fmt(u['v_c'])} | "
                f"{fmt(zh[0])} (ZH) | {fmt(zh[1])} | {fmt(r['free']['max_lean_deg'], 1)} |"
            )
        else:
            f = r["free"]
            row = (
                f"| {n}{mark} | {mass(r):.1f} | {fmt(u['v_w'])} | {fmt(u['v_c'])} | "
                f"{fmt(f['BT_V'])} | {fmt(f['BT_lean_deg'])} | {fmt(f['max_lean_deg'])} |"
            )
        lines.append(row)
    silver = measured_sets("riderless")["Silver"]
    lin = Linear(silver)
    sv = (lin.weave_speed((3.0, 6.0)), lin.capsize_speed())
    S["Silver upright (v_w, v_c)"] = sv
    lines.append(f"| Silver | -- | {fmt(sv[0])} | {fmt(sv[1])} | -- | -- | open |")

    # ---- Sect. 11: region and rule accuracy of the 23 sets --------------------------------------
    f = {n: R[n]["free"] for n in rider}
    S["tip speed range (m/s)"] = [
        min(x["BT_V"] for x in f.values()),
        max(x["BT_V"] for x in f.values()),
    ]
    S["tip lean range (deg)"] = [
        min(x["BT_lean_deg"] for x in f.values()),
        max(x["BT_lean_deg"] for x in f.values()),
    ]
    S["all 23 tips are BT points"] = all(x["tip_type"] == "BT" for x in f.values())
    S["all 23 boundaries otherwise stable"] = all(
        min(R[n][p]["valid_fraction"].values()) == 1.0 for n in rider for p in ("free", "constant")
    )
    no_yr = [n for n in rider if n != "Yellowrev+Jason"]
    lean = np.array([f[n]["max_lean_deg"] for n in no_yr])
    ay = np.array([f[n]["max_ay_g"] for n in no_yr])
    S["22 sets: largest lean range, median (deg)"] = [
        lean.min(),
        lean.max(),
        float(np.median(lean)),
    ]
    S["22 sets: largest a_y/g range"] = [ay.min(), ay.max()]
    below = [n for n in rider if f[n]["max_lean_deg"] < LIMIT]
    above = [n for n in rider if f[n]["max_lean_deg"] >= LIMIT]
    e2, e1 = rule_errors(R, below, "second_order"), rule_errors(R, below, "first_order")
    S["sets below 20 deg"] = len(below)
    S["second-order rule below 20 deg: rms, largest |error| (deg)"] = [
        rms(e2.values()),
        max(abs(v) for v in e2.values()),
    ]
    S["first-order rule below 20 deg: rms, largest error (deg), set"] = [
        rms(e1.values()),
        *signed_max(e1),
    ]
    S["first-order rule below 20 deg: overestimates"] = sum(v > 0 for v in e1.values())
    S["above 20 deg: second-order, first-order error (deg)"] = {
        n: [rule_errors(R, [n], "second_order")[n], rule_errors(R, [n], "first_order")[n]]
        for n in above
    }

    # ---- Sect. 11.2: Yellowrev with a rider -----------------------------------------------------
    yr = R["Yellowrev+Jason"]
    S["Yellowrev+Jason: v_c, tip V, tip lean, largest lean, at V, a_y/g"] = [
        yr["upright"]["v_c"],
        yr["free"]["BT_V"],
        yr["free"]["BT_lean_deg"],
        yr["free"]["max_lean_deg"],
        yr["free"]["V_at_max_lean"],
        yr["free"]["max_ay_g"],
    ]
    S["Yellowrev+Jason: tau3(v_c) (N m)"] = capsize_rule.coefficients(
        yr["params"], yr["upright"]["v_c"]
    )["tau3"]

    # ---- Sect. 8.3: speed of the tip ------------------------------------------------------------
    S["benchmark V_a3, BT speed (m/s)"] = [
        R["benchmark"]["upright"]["V_a3"],
        R["benchmark"]["free"]["BT_V"],
    ]
    q = [f[n]["BT_V"] / R[n]["upright"]["V_a3"] for n in rider]
    S["23 sets: V_BT / V_a3 range, free rolling"] = [min(q), max(q)]
    q = [R[n]["upright"]["V_a3"] / f[n]["BT_V"] for n in rider]
    S["23 sets: V_a3 / V_BT range, free rolling"] = [min(q), max(q)]
    qc = [R[n]["constant"]["BT_V"] / R[n]["upright"]["V_a3"] for n in rider]
    S["23 sets: largest |V_BT / V_a3 - 1|, constant speed"] = max(abs(x - 1) for x in qc)

    # ---- supplement, results for each set -------------------------------------------------------
    up = {n: R[n]["upright"] for n in rider}
    S["23 sets: v_w range"] = [
        min(u["v_w"] for u in up.values()),
        max(u["v_w"] for u in up.values()),
    ]
    S["23 sets: v_c range"] = [
        min(u["v_c"] for u in up.values()),
        max(u["v_c"] for u in up.values()),
    ]
    q = [f[n]["BT_V"] / up[n]["v_w"] for n in rider]
    S["23 sets: tip speed / v_w range"] = [min(q), max(q)]
    q = [R[n]["constant"]["max_lean_deg"] / f[n]["max_lean_deg"] - 1 for n in rider]
    S["23 sets: constant speed largest lean / free rolling - 1, range"] = [min(q), max(q)]
    tg = {n: R[n]["T_per_g@3"] for n in rider}
    adult = [n for n in rider if n not in ("Yellowrev+Jason", "Gyro+Child")]
    S["T/g at 3 m/s, adult riders (range, N m)"] = [
        max(tg[n] for n in adult),
        min(tg[n] for n in adult),
    ]
    S["T/g at 3 m/s, Yellowrev+Jason, Gyro+Child"] = [tg["Yellowrev+Jason"], tg["Gyro+Child"]]
    tl = [R[n]["T_per_g@3"] for n in riderless if "T_per_g@3" in R[n]]
    S["T/g at 3 m/s, riderless (range, N m)"] = [max(tl), min(tl)]
    per_kg = [abs(tg[n]) / mass(R[n]) for n in adult]
    S["|T/g| per kg of total mass, adult riders (range)"] = [min(per_kg), max(per_kg)]

    # countersteer on the grid of turns
    inside_ok, outside = True, {}
    for n, r in R.items():
        if "free" not in r or not math.isfinite(get(r, "free", "max_lean_deg")):
            continue
        L = r["free"]["max_lean_deg"]
        stable = [p for p in r["grid_free"] if p[5]]
        inside_ok &= all(p[4] < 0 for p in stable if p[0] <= L)
        out = [p for p in stable if p[0] > L + 1.0 and p[4] > 0]
        if out and n in rider:
            outside[n] = [min(p[0] for p in out), max(p[0] for p in out), min(p[3] for p in out)]
    S["grid: every stable turn up to the largest stable lean has T < 0"] = bool(inside_ok)
    S["grid, rider sets: stable turns with T > 0 beyond the region (lean min, max, a_y/g min)"] = (
        outside
    )

    # ---- supplement, effect of the rider --------------------------------------------------------
    pairs = [n for n in rider if R[n].get("partner") in R]
    S["pairs of the same bicycle with and without a rider"] = len(pairs)
    widen = [band(R[n]) / band(R[R[n]["partner"]]) for n in pairs]
    S["upright band widening factor (range)"] = [min(widen), max(widen)]
    both = [n for n in pairs if "free" in R[R[n]["partner"]]]
    grow = [
        n for n in both if R[n]["free"]["max_lean_deg"] > R[R[n]["partner"]]["free"]["max_lean_deg"]
    ]
    S["pairs with a region in both states; pairs where the largest lean grows"] = [
        len(both),
        len(grow),
    ]
    S["pairs where the largest lean falls (change, deg)"] = {
        n: R[n]["free"]["max_lean_deg"] - R[R[n]["partner"]]["free"]["max_lean_deg"]
        for n in both
        if n not in grow
    }
    dtip = {n: tip_V(R[n]) - tip_V(R[R[n]["partner"]]) for n in both}
    nb = [v for n, v in dtip.items() if R[n]["partner"] != "Browser"]
    S["tip speed rise, bicycles other than the Browser (range, m/s)"] = [min(nb), max(nb)]
    br = sorted(
        (mass(R[n]) - mass(R["Browser"]), R[n]["free"]["max_lean_deg"], R[n]["free"]["BT_V"], n)
        for n in rider
        if n.startswith("Browser+") and "Hess" not in n
    )
    S["Browser: rider mass (kg), largest lean (deg), tip speed (m/s)"] = br

    # ---- supplement, riderless measured bicycles ------------------------------------------------
    reg = [
        n
        for n in riderless
        if get(R[n], "free", "tip_type") == "BT"
        and min(R[n]["free"]["valid_fraction"].values()) == 1.0
    ]
    e1r = rule_errors(R, reg, "first_order")
    S["riderless with a benchmark-type region"] = sorted(reg)
    S["riderless: first-order rule rms, largest error (deg), set"] = [
        rms(e1r.values()),
        *signed_max(e1r),
    ]
    S["riderless: first-order rule underestimates"] = sum(v < 0 for v in e1r.values())

    # ---- tables of rule errors -------------------------------------------------------------------
    lines += [
        "\n## Capsize rules against the exact largest stable lean, free rolling (deg)\n",
        "| Set | exact | first-order rule - exact | second-order rule - exact |",
        "|---|---|---|---|",
    ]
    for n in sorted(rider, key=lambda s: R[s]["free"]["max_lean_deg"]):
        lines.append(
            f"| {n} | {fmt(f[n]['max_lean_deg'])} | "
            f"{fmt(rule_errors(R, [n], 'first_order')[n], sign=True)} | "
            f"{fmt(rule_errors(R, [n], 'second_order')[n], 3, sign=True)} |"
        )
    lines += ["\n## Stated numbers\n"]
    lines += [f"- {k}: {json.dumps(v, default=float)}" for k, v in S.items()]

    (DATA / "tables.md").write_text("\n".join(lines) + "\n")
    (DATA / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
