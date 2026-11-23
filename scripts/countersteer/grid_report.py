"""Grid counts for thm:countersteer (supplement, "Countersteer: proofs and tests")."""

import argparse
import json
import math

from common import RAW, write_json


def lines_of(o):
    out = {}
    for r in o["points"]:
        out.setdefault(round(r["V"], 9), []).append(r)
    return out


def neighbours(o, key):
    """Pairs of neighbouring turns on a speed line with a sign change of T, both stable."""
    for V, rows in lines_of(o).items():
        for a, b in zip(rows[:-1], rows[1:]):
            if a[key] and b[key] and a["T"] * b["T"] < 0 and abs(b["phi"] - a["phi"]) < 0.03:
                yield V, a, b


def report(o):
    vc, P = o["v_c"], o["points"]
    below_vc = (lambda r: True) if vc != vc else (lambda r: r["V"] < vc)
    S = [r for r in P if r["st_cs"]]
    inside = [r for r in S if r["below_first_fold_cs"] and below_vc(r)]
    cs = dict(
        stable=len(S),
        covered=len(inside),
        covered_T_neg=sum(r["T"] < 0 for r in inside),
        slope_neg=sum(r["T_phi_S"] < 0 for r in S),
    )
    F = [r for r in P if r["st_fr"] and "path_complete" in r]
    ok = [r for r in F if r["path_complete"] and r.get("path_min_a4", -1) > 0]
    out = [r for r in F if r not in ok]
    lo = [r for r in out if below_vc(r)]
    hi = [r for r in out if not below_vc(r)]
    fr = dict(
        tested=len(F),
        covered=len(ok),
        covered_T_neg=sum(r["T"] < 0 for r in ok),
        monotone=sum(r["path_T_monotone"] for r in ok),
        below_vc=len(lo),
        below_vc_T_pos=sum(r["T"] > 0 for r in lo),
        above_vc=len(hi),
        above_vc_T_pos=sum(r["T"] > 0 for r in hi),
        below_vc_lean_deg=[
            min((math.degrees(r["phi"]) for r in lo), default=None),
            max((math.degrees(r["phi"]) for r in lo), default=None),
        ],
        stable_hands_free=[
            (V, math.degrees(a["phi"]), math.degrees(b["phi"]))
            for V, a, b in neighbours(o, "st_fr")
        ],
    )
    # (d): first hands-free turn on each line below v_c has a4 < 0 at constant speed
    first_hf = first_hf_unstable = 0
    for V, rows in lines_of(o).items():
        if vc == vc and V >= vc:
            continue
        for a, b in zip(rows[:-1], rows[1:]):
            if a["T"] * b["T"] < 0 and abs(b["phi"] - a["phi"]) < 0.03:
                first_hf += 1
                first_hf_unstable += not (a["a4_cs"] > 0 and b["a4_cs"] > 0)
                break
    # (c): a fold of the constant-speed plant before each zero of T on a speed line
    zeros = no_fold = 0
    for V, rows in lines_of(o).items():
        ln = [x for x in o["lines"] if abs(x["V"] - V) < 1e-9]
        if not ln:
            continue
        s_prev, folds, T_prev = (1 if ln[0]["a40"] > 0 else -1), 0, None
        for a in rows:
            s = 1 if a["a4_cs"] > 0 else -1
            folds += s != s_prev
            s_prev = s
            if T_prev is not None and T_prev * a["T"] < 0:
                zeros += 1
                no_fold += folds == 0
                folds = 0
            T_prev = a["T"]
    L = [ln for ln in o["lines"] if ln.get("fold_cs") is not None and ln["a40"] > 0]
    both = [ln for ln in L if ln.get("fold_fr") is not None]
    gaps = [math.degrees(ln["fold_cs"] - ln["fold_fr"]) for ln in both]
    order = dict(
        lines=len(L),
        free_first=sum(g > 0 for g in gaps),
        both=len(both),
        gap_deg=[min(gaps, default=None), max(gaps, default=None)],
        exceptions=[(ln["V"], g) for ln, g in zip(both, gaps) if g <= 0],
        fold_vs_extremum_rad=max((abs(ln["T_ext_phi"] - ln["fold_cs"]) for ln in L), default=None),
        ratio_two_thirds=[ln["T_fold"] / (2 / 3 * ln["tau1"] * ln["fold_cs"]) for ln in L],
    )
    signs = {}
    for key, bad in (
        ("O1", lambda r: r["H_S"] <= 0),
        ("O2", lambda r: r["H_phi_S"] >= 0),
        ("O3", lambda r: r["T_S"] <= 0),
    ):
        B = [r for r in P if bad(r)]
        signs[key] = dict(
            n=len(B),
            stable=sum(r["st_cs"] or r["st_fr"] for r in B),
            lowest_lean_deg=min((math.degrees(r["phi"]) for r in B), default=None),
            below_first_fold=sum(r["below_first_fold_cs"] for r in B),
        )
    return dict(
        v_c=vc,
        v_w=o["v_w"],
        turns=len(P),
        max_Rpd=max(r["Rpd"] for r in P),
        max_Fcphi=max(r["Fcphi"] for r in P),
        constant=cs,
        free=fr,
        first_hands_free=first_hf,
        first_hands_free_unstable=first_hf_unstable,
        zeros_of_T=zeros,
        zeros_without_fold=no_fold,
        order=order,
        signs=signs,
        seconds=o.get("seconds"),
    )


def totals(per_set):
    t = dict(sets=len(per_set), turns=sum(s["turns"] for s in per_set.values()))
    t["max_Rpd"] = max(s["max_Rpd"] for s in per_set.values())
    t["max_Rpd_set"] = max(per_set, key=lambda k: per_set[k]["max_Rpd"])
    for plant in ("constant", "free"):
        keys = [k for k, v in next(iter(per_set.values()))[plant].items() if isinstance(v, int)]
        t[plant] = {k: sum(s[plant][k] for s in per_set.values()) for k in keys}
    for k in ("first_hands_free", "first_hands_free_unstable", "zeros_of_T", "zeros_without_fold"):
        t[k] = sum(s[k] for s in per_set.values())
    o = [s["order"] for s in per_set.values()]
    t["order"] = dict(
        lines=sum(x["lines"] for x in o),
        both=sum(x["both"] for x in o),
        free_first=sum(x["free_first"] for x in o),
        gap_max_deg=max(x["gap_deg"][1] for x in o if x["gap_deg"][1] is not None),
        gap_max_deg_without_long_trail=max(
            s["order"]["gap_deg"][1]
            for k, s in per_set.items()
            if not k.startswith("benchmark_c") and s["order"]["gap_deg"][1] is not None
        ),
        exceptions={
            k: s["order"]["exceptions"] for k, s in per_set.items() if s["order"]["exceptions"]
        },
        fold_vs_extremum_rad=max(x["fold_vs_extremum_rad"] or 0 for x in o),
    )
    t["signs"] = {
        k: dict(
            n=sum(s["signs"][k]["n"] for s in per_set.values()),
            sets=sum(s["signs"][k]["n"] > 0 for s in per_set.values()),
            stable=sum(s["signs"][k]["stable"] for s in per_set.values()),
            lowest_lean_deg=min(
                (
                    s["signs"][k]["lowest_lean_deg"]
                    for s in per_set.values()
                    if s["signs"][k]["lowest_lean_deg"] is not None
                ),
                default=None,
            ),
        )
        for k in ("O1", "O2", "O3")
    }
    t["seconds"] = sum(s["seconds"] or 0 for s in per_set.values())
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="report on the sets found, to results/")
    args = ap.parse_args()
    per_set = {}
    for f in sorted(RAW.glob("*.json")):
        if f.name.startswith("summary"):
            continue
        o = json.loads(f.read_text())
        per_set[o["name"]] = report(o)
    out = dict(totals=totals(per_set), sets=per_set)
    t = out["totals"]
    print(json.dumps({k: v for k, v in t.items() if k != "order"}, default=float))
    print(json.dumps(t["order"], default=float))
    if len(per_set) < 42 and not args.quick:
        raise SystemExit("run grid_scan.py for all 42 sets first, or use --quick")
    if args.quick:
        write_json("summary_quick.json", out, folder=RAW)
    else:
        write_json("grid_summary.json", out)


if __name__ == "__main__":
    main()
