"""Region of stable turns of each measured set (tab:measured-sets and tab:measured-riderless, Sect.
11, "Measured bicycles").
"""

# uv run python scripts/measured/analyse.py [names] [--force] [--out DIR]; about 1 min per set.

import argparse
import math
import time
from pathlib import Path

from common import (
    REPO,
    RESULTS,
    all_sets,
    grid_scan,
    region,
    rule_block,
    torque_per_g,
    upright,
    write_json_gz,
)

from whipple.model import Whipple


def analyse(name, p, group, partner):
    t0 = time.time()
    r = {"name": name, "group": group, "partner": partner, "params": p, "errors": []}
    bike = Whipple(p)
    try:
        r["upright"] = upright(p)
    except (RuntimeError, ValueError) as e:
        r["errors"].append(f"upright: {e!r}")
        r["upright"] = {"v_w": math.nan, "v_c": math.nan}
    for V in (3.0, 5.0):
        try:
            r[f"T_per_g@{V:g}"], note = torque_per_g(bike, p, V)
            if note:
                r.setdefault("notes", []).append(f"T_per_g@{V:g} {note}")
        except (RuntimeError, ValueError) as e:
            r["errors"].append(f"T/g@{V}: {e!r}")
    vw, vc = r["upright"].get("v_w", math.nan), r["upright"]["v_c"]
    for plant in ("free", "constant"):
        try:
            if not (vw == vw and vc == vc):
                raise RuntimeError("no upright weave or capsize speed")
            r[plant] = region(bike, plant, vw)
        except (RuntimeError, ValueError, IndexError, ArithmeticError) as e:
            r["errors"].append(f"{plant}: {e!r}")
    r["rule"] = rule_block(p)
    r["grid_free"] = grid_scan(bike, "free")
    r["seconds"] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="sets to compute (default: all)")
    ap.add_argument("--force", action="store_true", help="recompute sets that are stored")
    ap.add_argument("--out", type=Path, default=RESULTS, help="output folder")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO / args.out
    sets = all_sets()
    for name in args.names or list(sets):
        f = out / f"{name}.json.gz"
        if f.exists() and not args.force:
            continue
        r = analyse(name, *sets[name])
        write_json_gz(f, r)
        lean = r.get("free", {}).get("max_lean_deg", math.nan)
        print(
            f"{name}: {r['seconds']:.0f} s, largest stable lean (free rolling) {lean:.2f} deg, "
            f"{len(r['errors'])} errors",
            flush=True,
        )


if __name__ == "__main__":
    main()
