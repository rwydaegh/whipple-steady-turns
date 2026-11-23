"""tab:rule rows 1 to 4 and eq. H1 (supplement, "Capsize rule: order and accuracy")."""

import gzip
import json
import math

import numpy as np
from common import REPO, rule_fold_top, write_json

from whipple import capsize_rule as cr
from whipple.bicycles import BICYCLES, measured_sets

RESULTS = REPO / "data" / "measured" / "results"
NINE = [
    "Browserins",
    "Crescendo",
    "Fisher",
    "Gyro",
    "Pista",
    "Rigid",
    "Rigidcl",
    "Stratos",
    "Yellow",
]


def stored(name):
    f = RESULTS / f"{name}.json.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        return json.load(fh)


def row(name, p):
    r = stored(name)
    if r is not None and r.get("free"):
        exact = r["free"]["max_lean_deg"]
        exact_c = r["constant"]["max_lean_deg"]
        edge = r["free"]["max_lean_curve"]
    else:  # bicycle B is not among the stored results: exact fold top
        exact = math.degrees(rule_fold_top(p, "free").phi)
        exact_c = math.degrees(rule_fold_top(p, "constant").phi)
        edge = "fold"
    r2 = cr.largest_lean_rule(p, "free", 2)
    r1 = cr.largest_lean_rule(p, "free", 1)
    rc = cr.largest_lean_rule(p, "constant", 1)
    return dict(
        name=name,
        exact=exact,
        exact_constant=exact_c,
        edge=edge,
        valid=r2["valid"],
        second_order=r2["phi_deg"],
        V=r2["V"],
        first_order=r1["phi_deg"],
        constant_rule=rc["phi_deg"],
    )


def stats(rows, key, ref="exact"):
    e = np.array([r[key] - r[ref] for r in rows])
    return dict(
        rms=float(np.sqrt(np.mean(e**2))),
        largest=float(e[np.argmax(np.abs(e))]),
        overestimates=int(np.sum(e > 0)),
    )


def main():
    rider = {k: row(k, p) for k, p in measured_sets("rider").items()}
    riderless = {
        k: row(k, p) for k, p in measured_sets("riderless").items() if (stored(k) or {}).get("free")
    }  # Yellowrev has no capsize speed
    paper = {k: row(k, BICYCLES[k]) for k in ("benchmark", "A", "B")}
    below = [r for r in rider.values() if r["exact"] < 20]
    twelve = list(paper.values()) + [riderless[k] for k in NINE]
    table = {
        "measured bicycle and rider sets, lean below 20 deg": dict(
            n=len(below),
            exact=[min(r["exact"] for r in below), max(r["exact"] for r in below)],
            second_order=stats(below, "second_order"),
            first_order=stats(below, "first_order"),
        ),
        "Schwab et al. rider": rider["Rigid+rider(Schwab2013)"],
        "Yellowrev with rider": rider["Yellowrev+Jason"],
        "benchmark, A, B and nine riderless sets": dict(
            n=len(twelve),
            exact=[min(r["exact"] for r in twelve), max(r["exact"] for r in twelve)],
            second_order=stats(twelve, "second_order"),
            first_order=stats(twelve, "first_order"),
        ),
    }
    meet = [k for k, r in rider.items() if r["valid"] and r["edge"] == "fold"]
    out = dict(
        tab_rule=table,
        benchmark=paper["benchmark"],
        constant_rule_below_20deg=stats(below, "constant_rule", "exact_constant"),
        rider_sets_meeting_conditions=len(meet),
        riderless_failing_condition_c=[k for k, r in riderless.items() if r["edge"] != "fold"],
        rows=dict(**paper, **rider, **riderless),
    )
    write_json("rule_accuracy.json", out)
    b = paper["benchmark"]
    print(
        f"benchmark: second-order rule {b['second_order']:.2f} deg at {b['V']:.2f} m/s, "
        f"first-order {b['first_order']:.2f}, exact {b['exact']:.2f}"
    )
    for k, v in table.items():
        print(k, json.dumps(v, default=float)[:200])


if __name__ == "__main__":
    main()
