"""tab:rule row 5, random designs (supplement, "Capsize rule: order and accuracy")."""

import argparse
import math
import time

import numpy as np
from common import BICYCLES, fold_top, write_json

from whipple import capsize_rule as cr


def design(p):
    s = cr.series(p)
    row = dict(v_c=s.v_c, v_w=s.weave_speed(), direction=cr.speed_range(p)[2])
    if row["direction"] != "down":
        return row
    for plant in ("constant", "free"):
        r1, r2 = (cr.largest_lean_rule(p, plant, k) for k in (1, 2))
        c = s.coefficients(r2["V"])
        x0 = (r2["phi"], c["r1"] * r2["phi"] + c["r3"] * r2["phi"] ** 3)
        try:
            t = fold_top(s.bike, plant, x0)
            ok = abs(t.phi - r2["phi"]) < math.radians(3.0)
            exact, V = (math.degrees(t.phi), t.V) if ok else (None, None)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            exact, V = None, None
        row[plant] = dict(
            first_order=r1["phi_deg"], second_order=r2["phi_deg"], V=r2["V"], exact=exact, exact_V=V
        )
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="first 6 designs only")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(20261120)
    p0 = BICYCLES["benchmark"]
    rows = []
    for k in range(6 if args.quick else 40):
        p = {n: (v if n == "g" else v * math.exp(rng.uniform(-0.2, 0.2))) for n, v in p0.items()}
        rows.append(dict(k=k, **design(p)))
        f = rows[-1].get("free", {})
        print(k, f"exact {f.get('exact')}, second-order {f.get('second_order')}", flush=True)
        cr._CACHE.clear()
    summary = {"not ordinary": sum(r["direction"] != "down" for r in rows)}
    for plant in ("constant", "free"):
        good = [r[plant] for r in rows if r.get(plant, {}).get("exact")]
        ex = np.array([g["exact"] for g in good])
        summary[plant] = dict(n=len(good), exact=[ex.min(), ex.max()])
        for key in ("second_order", "first_order"):
            e = np.array([g[key] for g in good]) - ex
            summary[plant][key] = dict(rms=np.sqrt(np.mean(e**2)), largest=e[np.argmax(abs(e))])
    print(summary)
    write_json(
        "random_designs.json",
        dict(summary=summary, rows=rows, seconds=time.time() - t0),
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
