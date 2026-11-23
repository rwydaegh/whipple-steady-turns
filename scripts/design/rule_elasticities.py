"""Elasticities from the capsize rules (Sect. 12.2, "Design moves", fig:design-chart; supplement,
"Trade-offs with straight running")."""

import argparse
import json
import math

from common import BASES, DATA, write_json
from moves import LOADS, MOVES, with_rider

from whipple import capsize_rule as cr
from whipple.bicycles import BICYCLES

OUT = DATA / "rule_elasticities.json"
STEP, CM_STEP = 0.0025, 0.25
FIGURE_MOVES = (
    "trail", "front_spin_inertia", "rear_frame_forward", "rear_frame_mass", "rear_frame_higher",
    "total_mass", "wheelbase_only", "front_assembly_forward", "rear_wheel_back", "stretch",
    "front_radius", "front_radius_same_fork", "front_radius_same_frame", "head_angle",
    "head_angle_front_frame_rotates",
)  # fmt: skip
RANKING_MOVES = (
    "trail", "head_angle", "front_assembly_forward", "front_radius", "front_spin_inertia",
    "front_wheel_mass", "rear_frame_forward", "rear_frame_higher", "rear_frame_mass", "IByy",
    "IHyy", "IBxx", "front_frame_forward_of_axis",
)  # fmt: skip
RANKING_LOADS = ("Battery 3 kg on the down tube", "Child seat 20 kg",
                 "Basket 5 kg on the handlebar (front frame)", "Hub motor 3 kg, front")  # fmt: skip
# (key, plant, order): second-order rule for the figure, first-order rule for the ranking
RULES = {"free_second_order": ("free", 2), "free_first_order": ("free", 1),
         "constant_first_order": ("constant", 1)}  # fmt: skip


def top(p, plant, order):
    v = cr.largest_lean_rule(p, plant, order)["phi_deg"]
    cr._CACHE.clear()
    return v


def elasticity(p0, key, plant, order):
    move, kind, _ = MOVES[key]
    if kind == "pct":
        pp, pm, den = move(p0, math.exp(STEP)), move(p0, math.exp(-STEP)), 2 * STEP
    else:
        pp, pm, den = move(p0, CM_STEP), move(p0, -CM_STEP), 2 * CM_STEP / 100
    a, b = top(pp, plant, order), top(pm, plant, order)
    return (math.log(a) - math.log(b)) / den


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="figure moves of the benchmark only")
    a = ap.parse_args()
    res = {}
    jobs = [("benchmark", "free_second_order", FIGURE_MOVES)]
    if not a.quick:
        jobs += [("Crescendo+Jason", "free_second_order", FIGURE_MOVES)]
        jobs += [(b, r, RANKING_MOVES) for b in ("benchmark", "A_rider")
                 for r in ("free_first_order", "constant_first_order")]  # fmt: skip
    for base, rule, moves in jobs:
        p0 = BASES[base][0]
        plant, order = RULES[rule]
        r = res.setdefault(base, {}).setdefault(rule, {})
        r["base_deg"] = top(p0, plant, order)
        for key in moves:
            r[key] = elasticity(p0, key, plant, order)
            print(f"{base:16s} {rule:22s} {key:32s} {r[key]:+.4f}", flush=True)
        if moves is RANKING_MOVES:
            loads = dict(LOADS)
            if base == "A_rider":
                loads["Rider leans forward"] = lambda p: with_rider(BICYCLES["A"], 40.0)
            for name, load in loads.items():
                if name in RANKING_LOADS or name == "Rider leans forward":
                    r["load: " + name] = 100 * (top(load(p0), plant, order) / r["base_deg"] - 1)
    write_json(OUT, res)
    exact = json.loads((DATA / "elasticities.json").read_text())
    for base in ("benchmark", "Crescendo+Jason"):
        if base not in res or base not in exact:
            continue
        d = {k: res[base]["free_second_order"][k] - exact[base]["moves"][k]["lean_free"]
             for k in FIGURE_MOVES}  # fmt: skip
        k = max(d, key=lambda k: abs(d[k]))
        print(f"{base}: second-order rule minus exact, largest {d[k]:+.4f} ({k})")


if __name__ == "__main__":
    main()
