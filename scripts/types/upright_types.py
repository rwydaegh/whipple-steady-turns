"""tab:types, tau3 = P4(r) and the index v_c/V3 (supplement, "Bicycle types")."""

import json
import math

import numpy as np
from common import REPO, all_sets, inertia_admissible, poly_data, stored, write_json
from scipy.stats import spearmanr

from whipple.bicycles import measured_sets

RULE = REPO / "data" / "capsize" / "rule_accuracy.json"


def kind(o):
    if o["S_A"] <= 0:
        return "no stable turns"
    if not o["r_c"] > 0:
        return "no capsize speed"
    return "closed" if o["r_c"] > o.get("r3", math.inf) else "open"


def line_fit(x, y):
    co = np.polyfit(x, y, 1)
    err = np.polyval(co, x) - y
    loo = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        loo.append(np.polyval(np.polyfit(x[m], y[m], 1), x[i]) - y[i])
    loo = np.array(loo)
    return dict(
        slope=co[0],
        intercept=co[1],
        rms=np.sqrt(np.mean(err**2)),
        largest=np.max(np.abs(err)),
        worst=int(np.argmax(np.abs(err))),
        leave_one_out_rms=np.sqrt(np.mean(loo**2)),
        leave_one_out_largest=np.max(np.abs(loo)),
    )


def main():
    sets = all_sets()
    exact = json.loads(RULE.read_text())["rows"] if RULE.exists() else {}
    rows = {}
    for name, p in sets.items():
        o = poly_data(p)
        o["type"] = kind(o)
        o["admissible"] = inertia_admissible(p)
        r = stored(name)
        if r and r.get("constant"):
            o["lean_constant"] = r["constant"]["max_lean_deg"]
            o["lean_free"] = r["free"]["max_lean_deg"]
        elif name in exact:  # bicycle B: exact fold tops from the capsize rule script
            o["lean_constant"] = exact[name]["exact_constant"]
            o["lean_free"] = exact[name]["exact"]
        rows[name] = o
        print(
            f"{name:32s} {o['type']:16s} r_c {o['r_c']:7.4f} r3 {o.get('r3', math.nan):.4f} "
            f"V3 {o.get('V3', math.nan):.3f} v_c {o['v_c']:.3f} tau3(v_c) {o['tau3_vc']:+.4f} "
            f"P4 error {o['err_tau3']:.1e}",
            flush=True,
        )
    b = rows["benchmark"]
    rider = list(measured_sets("rider"))
    x = np.array([rows[k]["v_c"] / rows[k]["V3"] for k in rider])
    y = np.array([rows[k]["lean_constant"] for k in rider])
    ok = np.array([rows[k]["admissible"] for k in rider])
    fit = line_fit(x, y)
    fit["worst"] = rider[fit["worst"]]
    riderless_closed = [k for k in measured_sets("riderless") if rows[k]["type"] == "closed"]
    closed = {}
    for label, skip in (("all", ()), ("without Balanceassistv1", ("Balanceassistv1",))):
        names = [k for k in ("benchmark", "A", "B", *riderless_closed) if k not in skip]
        xc = np.array([rows[k]["v_c"] / rows[k]["V3"] for k in names])
        closed[label] = dict(
            n=len(names),
            spearman_constant=spearmanr(xc, [rows[k]["lean_constant"] for k in names]).statistic,
            spearman_free=spearmanr(xc, [rows[k]["lean_free"] for k in names]).statistic,
        )
    out = dict(
        benchmark=dict(
            coef=b["coef"],
            r_c=b["r_c"],
            r3=b["r3"],
            V3=b["V3"],
            v_c=b["v_c"],
            tau3_vc=b["tau3_vc"],
            V_r=b["V_r"],
        ),
        P4_largest_relative_error=max(o["err_tau3"] for o in rows.values()),
        P4_sets=len(rows),
        types={
            t: sorted(k for k, o in rows.items() if o["type"] == t)
            for t in ("closed", "open", "no capsize speed", "no stable turns")
        },
        riderless_closed=riderless_closed,
        index=dict(
            n=len(rider),
            spearman=spearmanr(x, y).statistic,
            spearman_admissible=spearmanr(x[ok], y[ok]).statistic,
            inadmissible=[k for k, g in zip(rider, ok) if not g],
            line=fit,
            closed=closed,
        ),
        rows=rows,
    )
    print({k: v for k, v in out["index"].items() if k != "line"}, fit)
    write_json("upright_types.json", out)


if __name__ == "__main__":
    main()
