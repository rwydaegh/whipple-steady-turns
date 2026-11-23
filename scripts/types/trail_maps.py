"""fig:types (c), maps at three trails (Sect. 10, "Closed and open regions")."""

import argparse
import json
import time

import numpy as np
from common import (
    BICYCLES,
    DATA,
    REPO,
    clean,
    hurwitz3,
    stability_map,
    stable_component,
    upright,
)

TRAILS = (0.080, 0.190, 0.220)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trails", nargs="*", type=float, help="trails in m (default: 0.08 0.19 0.22)")
    ap.add_argument("--quick", action="store_true", help="grid of 0.25 m/s and 1 deg")
    args = ap.parse_args()
    if args.quick:
        Vs, phis, folder = (
            np.arange(1.5, 14.01, 0.25),
            np.arange(1.0, 60.01, 1.0),
            REPO / "results/types",
        )
    else:
        Vs, phis, folder = np.arange(1.5, 14.01, 0.125), np.arange(0.5, 60.01, 0.5), DATA
    folder.mkdir(parents=True, exist_ok=True)
    for c in args.trails or TRAILS:
        t0 = time.time()
        p = dict(clean(BICYCLES["benchmark"]), c=c)
        u = upright(p)
        a, re = stability_map(p, Vs, phis)
        comp = stable_component(Vs, re, u["v_w"], u["v_c"])
        meta = dict(
            c=c,
            v_w=u["v_w"],
            v_c=u["v_c"],
            V3=u["V3"],
            D_free=u.get("D_free"),
            seconds=time.time() - t0,
        )
        f = folder / f"benchmark_c{c:.3f}.npz"
        np.savez_compressed(
            f,
            Vs=Vs,
            phis=phis,
            stable=comp,
            a4=a[..., 3].astype(np.float32),
            D3=hurwitz3(a).astype(np.float32),
            meta=json.dumps(meta),
        )
        top = phis[np.flatnonzero(comp.any(axis=0)).max()] if comp.any() else None
        print(
            f"c = {c:.3f} m: grid top {top} deg, {time.time() - t0:.0f} s; wrote",
            f.relative_to(REPO),
            flush=True,
        )


if __name__ == "__main__":
    main()
