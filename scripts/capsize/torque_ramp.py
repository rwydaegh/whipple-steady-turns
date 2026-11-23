"""fig:loading, torque limit and torque ramps (supplement, "Torque limit on a path")."""

import argparse
import math
import time

import numpy as np
from common import (
    BICYCLES,
    Whipple,
    clean,
    constant_speed_fold,
    first_fold_on_path,
    path,
    simulate_ramp,
    trim_grad,
    upright,
    write_json,
)

from whipple import capsize_rule as cr

V0 = 5.0
RATES = (0.004, 0.001)


def crossing(sol, level, n=20000):
    t = np.linspace(0, sol.t[-1], n)
    k = np.flatnonzero(sol.sol(t)[0] >= level)
    return t[k[0]] if len(k) else math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="the faster ramp only")
    args = ap.parse_args()
    t0 = time.time()
    p = clean(BICYCLES["benchmark"])
    bike = Whipple(p)
    lin = upright(p)["lin"]
    load = path(bike, lin, V0, "load", phi_end=0.45)
    fl = first_fold_on_path(bike, load)
    fh = first_fold_on_path(bike, path(bike, lin, V0, "H", phi_end=0.45))
    fc, Tc_constant = constant_speed_fold(bike, lin, V0)
    pred = dict(
        loading=dict(lean=fl["a4"], T=fl["T"], V=fl["V"], T_extremum_lean=fl["dT"]),
        H_curve=dict(lean=fh["a4"], T=fh["T"]),
        constant=dict(lean=fc, T=Tc_constant),
    )
    print(
        f"loading path: fold {math.degrees(fl['a4']):.3f} deg, T_c {fl['T']:.5f} N m, "
        f"V {fl['V']:.3f} m/s; constant speed: {math.degrees(fc):.3f} deg, {Tc_constant:.5f} N m"
    )
    sims, figure_ramps = [], []
    rates = RATES[:1] if args.quick else RATES
    for plant, phic, Tc in (("free", fl["a4"], fl["T"]), ("constant", fc, Tc_constant)):
        for rate in rates:
            sol = simulate_ramp(bike, V0, rate, plant, 3.0 * phic)
            t1, t2 = crossing(sol, phic), crossing(sol, 2 * phic)
            V1 = abs(sol.sol(t1)[4]) * p["rR"] if plant == "free" else V0
            sims.append(
                dict(
                    plant=plant,
                    rate=rate,
                    T_at_fold_lean=-rate * t1,
                    ratio_to_T_c=-rate * t1 / Tc,
                    T_at_twice_fold_lean=-rate * t2,
                    V_at_fold_lean=V1,
                )
            )
            print(
                f"{plant:8s} ramp {rate} N m/s: T at the fold lean {-rate * t1:.5f}, "
                f"ratio {-rate * t1 / Tc:.4f}, V {V1:.3f}",
                flush=True,
            )
            if plant == "free":
                s = simulate_ramp(bike, V0, rate, "free", 2.2 * fl["a4"])
                t = np.linspace(0, s.t[-1], 1500)
                z = s.sol(t)
                figure_ramps.append(dict(rate=rate, lean=z[0], T=-rate * t))
    # curves of fig:loading
    curve = []
    for x in np.linspace(load.t[0], load.t[-1], 400):
        v, _ = trim_grad(bike, x, float(load.sol(x)[0]))
        curve.append([x, v[1]])
    f = cr.series(p).fixed_H(V0)
    figure = dict(
        v0=V0,
        loading_path=curve,
        fold=pred["loading"],
        ramps=figure_ramps,
        series=dict(tau1=f["tau1"], tau3H=f["tau3H"], tau5H=f["tau5H"]),
    )
    out = dict(v0=V0, prediction=pred, ramps=sims, seconds=time.time() - t0)
    write_json("torque_ramp.json", out, quick=args.quick)
    if not args.quick:
        for r in figure["ramps"]:
            r["lean"], r["T"] = np.round(r["lean"], 7).tolist(), np.round(r["T"], 7).tolist()
        figure["loading_path"] = np.round(figure["loading_path"], 9).tolist()
        write_json("loading_path_figure.json", figure)


if __name__ == "__main__":
    main()
