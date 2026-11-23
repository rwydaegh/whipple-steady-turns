"""Tables and numbers of the supplement section "Random designs" from the results file."""

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "random_designs" / "results.jsonl.gz"
GOOD_TOP = ("newton", "tip", "second pass")


def g(d, *ks, default=math.nan):
    for k in ks:
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def load(path=RESULTS):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def closed_or_open(t):
    return "open" if t == "open" else "closed"


def types(R):
    """Exact types of the random designs, and the closed/open type against D_free at v_c."""
    out = {"exact": dict(Counter(g(r["result"], "free", "type", default="missing") for r in R))}
    bins = [(-math.inf, 0), (0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1), (1, math.inf)]
    sub = [r for r in R if g(r["result"], "free", "type", default="") in
           ("closed_BT", "closed_zH", "open")]  # fmt: skip
    rows = []
    for lo, hi in bins:
        S = [r for r in sub if lo <= g(r["result"], "pred", "D_free") < hi]
        n_open = sum(g(r["result"], "free", "type") == "open" for r in S)
        rows.append(dict(lo=lo, hi=hi, n=len(S), open=n_open, closed=len(S) - n_open))
    out["D_free"] = rows
    out["open_predicted_closed"] = sum(
        g(r["result"], "free", "type") == "open"
        and g(r["result"], "pred", "free", "type", default="").startswith("closed")
        for r in sub
    )
    return out


def rule_errors(A):
    """Errors of the first-order and second-order rules for the largest stable lean."""
    C = [r for r in A if g(r["result"], "free", "type", default="").startswith("closed")
         and g(r["result"], "free", "top_method", default="") in GOOD_TOP]  # fmt: skip

    def err(r, form):
        return g(r["result"], "pred", "free", form) - g(r["result"], "free", "top_deg")

    def valid(r):
        q = g(r["result"], "pred", "free", default={})
        return (
            bool(q.get("first_order_valid"))
            and q.get("direction") == "down"
            and g(q, "weave_margin") < 0
        )

    V = [r for r in C if valid(r)]
    out = {"n_conditions_hold": len(V), "bins": []}
    for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 90)):
        S = [r for r in V if lo <= g(r["result"], "free", "top_deg") < hi]
        e1 = np.abs([err(r, "first_order") for r in S])
        e2 = np.abs([err(r, "second_order") for r in S])
        out["bins"].append(dict(lo=lo, hi=hi, n=len(S), first_median=float(np.nanmedian(e1)),
                                first_p90=float(np.nanpercentile(e1, 90)),
                                second_median=float(np.nanmedian(e2)),
                                second_p90=float(np.nanpercentile(e2, 90))))  # fmt: skip
    small = [r for r in V if g(r["result"], "free", "top_deg") <= 20]
    e = np.array([err(r, "first_order") for r in small])
    out["n_top_le_20"] = len(small)
    out["fraction_too_high_le_20"] = float(np.mean(e[np.isfinite(e)] > 0))
    e = np.array([err(r, "first_order") for r in V])
    out["fraction_too_high_all"] = float(np.mean(e[np.isfinite(e)] > 0))
    e = np.abs(e)
    out["first_p90_all"] = float(np.nanpercentile(e, 90))
    out["first_fail_conditions_hold"] = int(np.sum(~(e <= 1.0)))
    for lab, lo, hi in (("differ_lt_0.3", 0, 0.3), ("differ_gt_1_or_none", 1.0, math.inf)):
        S = []
        for r in V:
            d = abs(g(r["result"], "pred", "free", "first_order")
                    - g(r["result"], "pred", "free", "second_order"))  # fmt: skip
            d = math.inf if not np.isfinite(d) else d
            if lo <= d < hi:
                S.append(r)
        ec = np.abs([err(r, "first_order") for r in S])
        out[lab] = dict(n=len(S), first_error_gt_1=int(np.sum(~(ec <= 1.0))))
    return out


def tip_and_weave(A):
    """V_a3 against the exact tip speed, and the hybrid weave side, closed regions."""
    B = [r for r in A if g(r["result"], "free", "tip_kind", default="") == "BT"
         and g(r["result"], "free", "type", default="").startswith("closed")]  # fmt: skip
    rel = np.array([100 * (g(r["result"], "pred", "V_a3") / g(r["result"], "free", "tip_V") - 1)
                    for r in B])  # fmt: skip
    rel = rel[np.isfinite(rel)]
    CC = [r for r in A if g(r["result"], "free", "type", default="").startswith("closed")]
    w = np.abs([g(r["result"], "free", "err_hopf_hyb") for r in CC])
    w = w[np.isfinite(w)]
    return dict(
        n_bt=len(B),
        V_a3_above=float(np.mean(rel > 0)),
        V_a3_abs=[float(np.percentile(np.abs(rel), q)) for q in (50, 90, 99)],
        n_weave=len(w),
        weave_abs=[float(np.percentile(w, q)) for q in (50, 90, 99)],
    )


def main():
    rows = load()
    R = [r for r in rows if r["kind"] == "rand"]
    A = [r for r in rows if r["kind"] in ("rand", "anchor")]
    secs = [r["seconds"] for r in R]
    print(
        f"{len(R)} random designs and {len(A) - len(R)} anchors. CPU time of the random "
        f"designs {sum(secs) / 3600:.1f} h, median {np.median(secs):.0f} s per design."
    )
    t = types(R)
    print(
        "Exact types (free rolling):", ", ".join(f"{k} {v}" for k, v in sorted(t["exact"].items()))
    )
    print(f"Exact open, D_free > 0 (predicted closed): {t['open_predicted_closed']}")
    print("\nTypes\n| D_free (N m) | Designs | Exact open | Exact closed |")
    print("|---|---|---|---|")
    for b in t["D_free"]:
        lab = (
            "< 0"
            if b["lo"] == -math.inf
            else f">= {b['lo']:g}"
            if b["hi"] == math.inf
            else f"{b['lo']:g} to {b['hi']:g}"
        )
        print(f"| {lab} | {b['n']} | {b['open']} | {b['closed']} |")
    e = rule_errors(A)
    print(f"\nRule errors ({e['n_conditions_hold']} designs meet the three conditions)")
    print(
        "| Exact lean (deg) | Designs | first median | first 90th | second median | second 90th |"
    )
    print("|---|---|---|---|---|---|")
    for b in e["bins"]:
        lab = f"above {b['lo']}" if b["hi"] == 90 else f"{b['lo']} to {b['hi']}"
        print(
            f"| {lab} | {b['n']} | {b['first_median']:.2f} | {b['first_p90']:.2f} | "
            f"{b['second_median']:.2f} | {b['second_p90']:.2f} |"
        )
    print(
        f"\nTop <= 20 deg: {e['n_top_le_20']} designs, first-order rule too high in "
        f"{100 * e['fraction_too_high_le_20']:.0f} % (all sizes "
        f"{100 * e['fraction_too_high_all']:.0f} %). First-order 90th percentile, all sizes: "
        f"{e['first_p90_all']:.2f} deg. Error > 1 deg with all conditions: "
        f"{e['first_fail_conditions_hold']} of {e['n_conditions_hold']}."
    )
    for k in ("differ_lt_0.3", "differ_gt_1_or_none"):
        print(
            f"Rules {k}: {e[k]['n']} designs, first-order error > 1 deg in "
            f"{e[k]['first_error_gt_1']}."
        )
    w = tip_and_weave(A)
    print(
        f"V_a3 on {w['n_bt']} closed regions with a BT tip: above the tip in "
        f"{100 * w['V_a3_above']:.0f} %, |error| median / 90th / 99th "
        + " / ".join(f"{x:.1f}" for x in w["V_a3_abs"])
        + " %."
    )
    print(
        f"Hybrid Hopf curve, largest lean error ({w['n_weave']} designs): median / 90th / 99th "
        + " / ".join(f"{x:.3f}" for x in w["weave_abs"])
        + " deg."
    )


if __name__ == "__main__":
    main()
