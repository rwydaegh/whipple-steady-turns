"""Torque limit on 68 paths, eq. twothirds (supplement, "Torque limit on a path")."""

import argparse
import json
import math
import time

import numpy as np
from common import (
    REPO,
    Whipple,
    all_sets,
    constant_speed_fold,
    first_fold_on_path,
    path,
    upright,
    write_json,
)

SETS = [
    "benchmark",
    "A",
    "B",
    "Browser+Jason",
    "Browserins+Jason",
    "Stratos+Jason",
    "Rigid+Jason",
    "Balanceassistv1+Jason",
    "Rigidcl+Luke",
    "Rigidcl+Charlie",
    "Gyro+Child",
    "Crescendo+Jason",
    "Yellow+Jason",
    "Fisher+Jason72(Hess2012)",
    "Pista+Jason72(Hess2012)",
    "Rigid+rider(Schwab2013)",
    "DelftBrowser+Arend(Moore2012)",
]


def run(name, p):
    bike = Whipple(p)
    u = upright(p)
    lin, vc, vw = u["lin"], u["v_c"], u["v_w"]
    rows = []
    for frac in (0.35, 0.7):
        v0 = vw + frac * (vc - vw)
        row = dict(name=name, v0=v0, v_c=vc, v_w=vw, tau1=lin.tau1(v0))
        for kind in ("H", "load"):
            row[kind] = first_fold_on_path(bike, path(bike, lin, v0, kind))
        fc, Tc = constant_speed_fold(bike, lin, v0)
        row["constant"] = dict(a4=fc, T=Tc)
        rows.append(row)
        print(
            f"{name:30s} v0 {v0:6.3f}: loading path fold {math.degrees(row['load']['a4']):.4f} "
            f"deg, T_c {row['load']['T']:.4f}; constant speed {math.degrees(fc):.4f}, {Tc:.4f}",
            flush=True,
        )
    return rows


def summary(rows):
    ratio = [
        r[k]["T"] / (2 / 3 * r["tau1"] * r[k]["a4"]) for r in rows for k in ("load", "constant")
    ]
    return dict(
        paths=len(ratio),
        sets=len({r["name"] for r in rows}),
        ratio_two_thirds=[min(ratio), max(ratio)],
        H_path_fold_vs_extremum_rad=max(abs(r["H"]["a4"] - r["H"]["dT"]) for r in rows),
        loading_fold_vs_extremum_rad=max(abs(r["load"]["a4"] - r["load"]["dT"]) for r in rows),
        loading_vs_H_fold_rad=max(abs(r["load"]["a4"] - r["H"]["a4"]) for r in rows),
        free_over_constant_limit=[
            min(r["load"]["T"] / r["constant"]["T"] for r in rows),
            max(r["load"]["T"] / r["constant"]["T"] for r in rows),
        ],
        constant_minus_free_lean_deg=[
            min(math.degrees(r["constant"]["a4"] - r["load"]["a4"]) for r in rows),
            max(math.degrees(r["constant"]["a4"] - r["load"]["a4"]) for r in rows),
        ],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="sets to compute now (default: all 17)")
    ap.add_argument("--quick", action="store_true", help="three sets only")
    args = ap.parse_args()
    sets = all_sets()
    names = SETS[:3] if args.quick else (args.names or SETS)
    part = REPO / "results" / "capsize" / "loading_paths"
    part.mkdir(parents=True, exist_ok=True)
    for n in names:
        t0 = time.time()
        rows = run(n, sets[n])
        (part / f"{n}.json").write_text(
            json.dumps(dict(rows=rows, seconds=time.time() - t0), default=float)
        )
    done = {
        n: json.loads((part / f"{n}.json").read_text())
        for n in SETS
        if (part / f"{n}.json").exists()
    }
    rows = [r for n in (names if args.quick else done) for r in done[n]["rows"]]
    out = dict(summary=summary(rows), paths=rows, seconds=sum(done[n]["seconds"] for n in done))
    print({k: np.round(v, 9).tolist() for k, v in out["summary"].items()})
    if args.quick or len(done) == len(SETS):
        write_json("loading_paths.json", out, quick=args.quick)
    else:
        print(f"{len(done)} of {len(SETS)} sets done; run the others to write the summary")


if __name__ == "__main__":
    main()
