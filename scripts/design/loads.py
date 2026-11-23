"""Added loads on the benchmark bicycle, tab:design-loads (supplement, "Mass of the rider and
position of the rider's mass centre")."""

import argparse

from common import DATA, Base, write_json
from moves import LOADS, with_rider

from whipple.bicycles import BICYCLES

OUT = DATA / "loads.json"
NSUB = 6  # parameter substeps from the base set to the loaded set


def evaluate(base, p1):
    """Continue the fold tops and the tip from the base set to p1 in NSUB linear substeps."""
    guess = dict(base.guess)
    for k in range(1, NSUB + 1):
        pk = {q: base.p[q] + k / NSUB * (p1[q] - base.p[q]) for q in base.p}
        v = Base.evaluate(base, pk, ("lean_free", "lean_constant", "tip_V"))
        for key in ("free", "constant"):
            base.guess[key] = v[f"x_{key}"]
        base.guess["tip"] = v["x_tip"]
    base.guess = guess
    v.update(Base.evaluate(base, p1, ("v_w", "v_c")))
    return {k: v[k] for k in ("v_w", "v_c", "lean_free", "lean_constant", "tip_V", "ok_free")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="benchmark only")
    a = ap.parse_args()
    res = {}
    for name in ["benchmark"] + ([] if a.quick else ["A_rider"]):
        base = Base(name)
        loads = dict(LOADS)
        if name == "A_rider":
            loads["Rider leans forward"] = lambda p: with_rider(BICYCLES["A"], 40.0)
        rows = {"None": {k: base.values[k] for k in ("v_w", "v_c", "lean_free", "lean_constant",
                                                     "tip_V")}}  # fmt: skip
        for label, load in loads.items():
            rows[label] = evaluate(base, load(base.p))
        res[name] = rows
        print(f"\n{name}\n| Load | v_w (m/s) | phi* (deg) | V_tip (m/s) |\n|---|---|---|---|")
        for label, r in rows.items():
            print(f"| {label} | {r['v_w']:.2f} | {r['lean_free']:.2f} | {r['tip_V']:.2f} |")
    write_json(OUT, res)


if __name__ == "__main__":
    main()
