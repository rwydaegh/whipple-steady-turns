"""Elasticities of the design moves, tab:design-chart, tab:design-mass, tab:design-variants,
tab:design-tradeoff (Sect. 12.2-12.3; supplement, "Design moves")."""

import argparse
import json
import math
import time

from common import BASE_LABELS, BASES, DATA, Base, elasticity, write_json
from moves import MOVES

OUT = DATA / "elasticities.json"
CM_STEP = 0.25  # cm, position moves
QUICK_MOVES = ("trail", "front_spin_inertia", "rear_frame_forward", "total_mass")

# (move key, row label) of each table; the bases of each table
CHART = [
    ("trail", "Trail c"), ("head_angle", "Head angle, slacker*"),
    ("front_spin_inertia", "Front wheel spin inertia"), ("front_wheel_mass", "Front wheel mass"),
    ("front_radius", "Front wheel radius*"),
    ("rear_frame_forward", "Rear-frame mass centre forward"),
    ("rear_frame_mass", "Rear-frame mass, same build"),
    ("front_assembly_forward", "Front assembly forward*"),
    ("rear_frame_higher", "Rear-frame mass centre higher"), ("IByy", "Rear-frame pitch inertia"),
    ("front_frame_mass", "Front-frame mass"), ("rear_radius", "Rear wheel radius*"),
    ("total_mass", "All masses and inertias"),
    ("front_frame_forward_of_axis", "Front-frame centre forward of the axis (% per cm)"),
    ("front_frame_up_axis", "Front-frame centre up the axis (% per cm)"),
]  # fmt: skip
OTHER_INERTIAS = ("IHyy", "IBxx", "IBzz", "IBxz", "IHxx", "IHzz", "IHxz", "IFxx", "IRxx")
MASS = [
    ("rear_frame_mass_only", "Mass only"), ("rear_frame_mass", "Same build"),
    ("rear_frame_taller", "Taller rider"), ("rear_frame_forward", "Centre forward"),
    ("rear_frame_higher", "Centre higher"), ("total_mass", "All masses and inertias"),
]  # fmt: skip
VARIANTS = [
    ("wheelbase_only", "W1: w only"), ("front_assembly_forward", "W2: front assembly forward"),
    ("rear_wheel_back", "W3: rear wheel back"), ("stretch", "W4: stretch of w, xB, xH"),
    ("front_radius", "rF: inertia with rF^2"), ("front_radius_geometry_only", "rF: geometry only"),
    ("front_radius_inertia_only", "rF: inertia only"), ("front_radius_same_fork", "rF: same fork"),
    ("front_radius_same_frame", "rF: same frame, pitched"), ("rear_radius", "rR: no pitch"),
    ("rear_radius_same_frame", "rR: same frame, pitched"),
    ("head_angle", "Head angle: front-frame centre fixed"),
    ("head_angle_front_frame_rotates", "Head angle: front frame rotates"),
    ("trail", "Trail: front-frame centre fixed"),
    ("trail_with_front_frame", "Trail: front-frame centre with the axis"),
]  # fmt: skip
TRADEOFF = [
    ("trail", "Trail c"), ("head_angle", "Head angle, slacker (front-frame centre fixed)"),
    ("front_spin_inertia", "Front wheel spin inertia"),
    ("front_frame_forward_of_axis", "Front-frame centre forward of the axis"),
    ("rear_frame_forward", "Rear-frame mass centre forward"),
    ("front_assembly_forward", "Front assembly forward (W2)"),
    ("rear_frame_higher", "Rear-frame mass centre higher"),
]  # fmt: skip


def compute(names, step, moves):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        t0 = time.time()
        base = Base(name)
        r = res.setdefault(name, {})
        r["base"] = {k: v for k, v in base.values.items() if not k.startswith("x_")}
        r["step"], r["cm_step"] = step, CM_STEP
        el = r.setdefault("moves", {})
        for key in moves:
            move, kind, _ = MOVES[key]
            el[key] = elasticity(base, move, kind, step if kind == "pct" else CM_STEP)
            e = el[key]
            print(f"{name:16s} {key:32s} lean free {e['lean_free']:+.3f}  constant "
                  f"{e['lean_constant']:+.3f}  tip {e['tip_V']:+.3f}  v_w {e['v_w']:+.3f}  "
                  f"v_c {e['v_c']:+.3f}  T3 {e['T3']:+.3f}  {'' if e['ok'] else 'CHECK'}",
                  flush=True)  # fmt: skip
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
        write_json(OUT, res)
    return res


def fmt(res, base, key, q="lean_free", nd=2):
    try:
        v = res[base]["moves"][key][q]
    except KeyError:
        return "--"
    v = 0.0 if abs(v) < 0.5 * 10**-nd else v
    return f"{v:+.{nd}f}"


def table(title, rows, bases, res, q="lean_free"):
    print(f"\n{title}\n")
    print("| Move | " + " | ".join(BASE_LABELS[b] for b in bases) + " |")
    print("|---" * (len(bases) + 1) + "|")
    for key, label in rows:
        nd = 1 if MOVES[key][1] == "cm" and key.endswith("of_axis") else 2
        print(f"| {label} | " + " | ".join(fmt(res, b, key, q, nd) for b in bases) + " |")


def tables(res):
    have = [b for b in BASES if b in res]
    print(
        "Base values of phi* (deg, free rolling): "
        + ", ".join(f"{BASE_LABELS[b]} {res[b]['base']['lean_free']:.2f}" for b in have)
    )
    three = [b for b in ("benchmark", "Crescendo+Jason", "Browser+Jason") if b in res]
    table("tab:design-chart (elasticity of phi*, free rolling)", CHART, three, res)
    worst = max((abs(res[b]["moves"][k]["lean_free"]) for b in three for k in OTHER_INERTIAS
                 if k in res[b]["moves"]), default=math.nan)  # fmt: skip
    print(f"| Other single inertias {', '.join(OTHER_INERTIAS)} | max |eps| = {worst:.3f} |")
    two = [b for b in ("benchmark", "Crescendo+Jason") if b in res]
    table("tab:design-mass (elasticity of phi*, free rolling)", MASS, two, res)
    table("tab:design-variants (elasticity of phi*, free rolling)", VARIANTS, three, res)
    tb = [b for b in ("benchmark", "A_rider") if b in res]
    for q, lab in (("lean_free", "phi*"), ("v_w", "v_w"), ("tip_V", "V_tip")):
        table(f"tab:design-tradeoff, {lab} (benchmark / A + rider)", TRADEOFF, tb, res, q)
    print("\nStatements of the text (all bases computed):")
    for key, q in (
        ("trail", "tip_V"),
        ("trail", "T3"),
        ("trail_with_front_frame", "lean_free"),
        ("rear_frame_forward", "v_w"),
        ("rear_frame_forward", "tip_V"),
        ("head_angle_front_frame_rotates", "v_c"),
        ("IByy", "lean_free"),
        ("IHyy", "lean_free"),
        ("front_frame_forward_of_axis", "lean_free"),
    ):
        vals = ", ".join(f"{BASE_LABELS[b]} {fmt(res, b, key, q)}" for b in have)
        print(f"  {key} -> {q}: {vals}")
    roll_yaw = ("IBxx", "IBzz", "IBxz", "IHxx", "IHzz", "IHxz", "IFxx")
    if "benchmark" in res:
        m = res["benchmark"]["moves"]
        print(
            "  benchmark roll/yaw inertias: max |eps(v_w)| = "
            f"{max((abs(m[k]['v_w']) for k in roll_yaw if k in m), default=math.nan):.3f}, "
            f"max |eps(phi*)| = "
            f"{max((abs(m[k]['lean_free']) for k in roll_yaw if k in m), default=math.nan):.3f}"
        )
    rider = [b for b in ("benchmark", "A_rider", "Crescendo+Jason", "Browser+Jason") if b in res]
    dif = max(
        abs(res[b]["moves"][k]["lean_constant"] - res[b]["moves"][k]["lean_free"])
        for b in rider
        for k in res[b]["moves"]
    )
    print(
        f"  largest |eps(constant speed) - eps(free rolling)| over all moves and bases: {dif:.2f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", nargs="*", default=list(BASES), choices=list(BASES))
    ap.add_argument("--step", type=float, default=0.0025, help="log step of percentage moves")
    ap.add_argument("--quick", action="store_true", help="benchmark and four moves only")
    ap.add_argument("--tables", action="store_true", help="print the tables from the stored file")
    a = ap.parse_args()
    if not a.tables:
        names = ["benchmark"] if a.quick else a.base
        moves = QUICK_MOVES if a.quick else list(MOVES)
        compute(names, a.step, moves)
    res = json.loads(OUT.read_text())
    if not all(math.isfinite(res[b]["base"]["lean_free"]) for b in res):
        print("warning: a base value is not finite")
    tables(res)


if __name__ == "__main__":
    main()
