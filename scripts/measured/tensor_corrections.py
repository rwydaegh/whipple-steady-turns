"""Inadmissible front-frame inertia tensors and the effect of their correction (tab:tensors, Sect.
"Inadmissible front-frame inertia tensors").
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from common import DATA, excess, load_result, region, rule_block, tensor, upright

from whipple.bicycles import measured_sets
from whipple.model import Whipple

OUT = DATA / "tensor_corrections.json"
OUT_SIGMA = DATA / "tensor_uncertainty.json"


def project(inertia):
    """Nearest admissible tensor: same principal axes, I_max lowered, the others raised by e/3."""
    lam, V = np.linalg.eigh(inertia)
    e = lam[2] - lam[1] - lam[0]
    if e <= 0:
        return inertia.copy()
    return V @ np.diag(lam + np.array([e / 3, e / 3, -e / 3])) @ V.T


def with_tensor(p, body, inertia):
    q = dict(p)
    q[f"I{body}xx"], q[f"I{body}yy"], q[f"I{body}zz"], q[f"I{body}xz"] = (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 2],
    )
    return q


def summary(up, free, rule):
    return dict(
        v_w=up["v_w"],
        v_c=up["v_c"],
        V_a3=up["V_a3"],
        tipV_free=free["BT_V"],
        tiplean_free=free["BT_lean_deg"],
        lean_free=free["max_lean_deg"],
        V_lean_free=free["V_at_max_lean"],
        rule_free=rule["free_first_order"]["phi_deg"],
        rule_free_second=rule["free_second_order"]["phi_deg"],
    )


def evaluate(p):
    up = upright(p)
    return summary(up, region(Whipple(p), "free", up["v_w"]), rule_block(p))


def rule_stats(rows, key, limit=20.0):
    e = [(r[key] - r["lean_free"], n) for n, r in rows.items() if r["lean_free"] < limit]
    a = np.array([x[0] for x in e])
    k = int(np.argmax(np.abs(a)))
    return dict(n=len(a), rms=float(np.sqrt(np.mean(a**2))), max_signed=float(a[k]), worst=e[k][1])


def corrections():
    sets = {"rider": measured_sets("rider"), "riderless": measured_sets("riderless")}
    bad = []
    for group, d in sets.items():
        for n, p in d.items():
            for b in "BH":
                e, lam = excess(tensor(p, b))
                if e > 0:
                    bad.append(
                        dict(
                            group=group,
                            set=n,
                            body=b,
                            excess=e,
                            moments=lam.tolist(),
                            rel=e / lam[2],
                        )
                    )
    five = [a["set"] for a in bad if a["group"] == "rider"]
    rider = sets["rider"]
    variants = {}
    for n in five:
        p = rider[n]
        variants[f"{n}|projected"] = with_tensor(p, "H", project(tensor(p, "H")))
        if n.startswith("DelftBrowser"):  # assumed IHyy raised to the smallest admissible value
            lam = np.linalg.eigvalsh(np.array([[p["IHxx"], p["IHxz"]], [p["IHxz"], p["IHzz"]]]))
            variants[f"{n}|IHyy raised"] = dict(p, IHyy=float(lam[1] - lam[0]))
        if n == "Yellow+Jason72(Hess2012)":  # IHzz of the measured Yellow instead of 0.023
            variants[f"{n}|IHzz=0.029"] = dict(p, IHzz=0.029)
    out = {}
    for key, q in variants.items():
        n = key.split("|")[0]
        r = load_result(n)
        base = summary(r["upright"], r["free"], r["rule"])
        new = evaluate(q)
        out[key] = dict(
            base=base,
            fixed=new,
            delta={k: new[k] - base[k] for k in base},
            H_after=[q[k] for k in ("IHxx", "IHyy", "IHzz", "IHxz")],
            excess_after=excess(tensor(q, "H"))[0],
        )
        print(key, {k: round(v, 4) for k, v in out[key]["delta"].items()}, flush=True)
    rows = {
        n: summary(*(lambda r: (r["upright"], r["free"], r["rule"]))(load_result(n))) for n in rider
    }
    proj = dict(rows, **{n: out[f"{n}|projected"]["fixed"] for n in five})
    stats = {
        "stored": {k: rule_stats(rows, k) for k in ("rule_free", "rule_free_second")},
        "projected": {k: rule_stats(proj, k) for k in ("rule_free", "rule_free_second")},
    }
    OUT.write_text(
        json.dumps(dict(inadmissible=bad, corrected=out, rule_stats_below_20deg=stats), indent=1)
        + "\n"
    )
    print(json.dumps(stats, indent=1))


def uncertainty(data, scratch):
    """Triangle excess with its standard deviation from the raw measurements (bicycleparameters)."""
    import bicycleparameters as bp
    from uncertainties import umath

    work = Path(scratch) / "data"
    if not work.exists():
        shutil.copytree(data, work)  # the package writes into its data folder
    res = {}
    for name in measured_sets("riderless"):
        try:
            P = bp.Bicycle(name, pathToData=str(work), forceRawCalc=True).parameters["Benchmark"]
        except (KeyError, OSError, ValueError) as e:
            res[name] = {"error": repr(e)}
            continue
        res[name] = {}
        for b in "BH":
            xx, yy, zz, xz = (P[f"I{b}{k}"] for k in ("xx", "yy", "zz", "xz"))
            r = umath.sqrt(((xx - zz) / 2) ** 2 + xz**2)
            lp, lm = (xx + zz) / 2 + r, (xx + zz) / 2 - r
            e = max(yy - lp - lm, lp - yy - lm, key=lambda u: u.nominal_value)
            res[name][b] = dict(excess=e.nominal_value, sigma=e.std_dev)
    OUT_SIGMA.write_text(json.dumps(res, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
    ap.add_argument(
        "--uncertainty",
        nargs=2,
        metavar=("DATA", "SCRATCH"),
        help="instead: excess with uncertainty from a copy of the BicycleParameters data "
        "folder (needs bicycleparameters 1.5.2)",
    )
    args = ap.parse_args()
    if args.uncertainty:
        uncertainty(*args.uncertainty)
    else:
        corrections()


if __name__ == "__main__":
    main()
