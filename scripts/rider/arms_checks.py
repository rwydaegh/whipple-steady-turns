"""Checks of the spring formulas, eq. rider-kc and Proposition prop:rider-fold(i), against the full model (Sect. 13.2; supplementary material, "Constant speed and a numerical test of the capsize condition" and "Centring spring: further results")."""  # noqa: E501

import json
import math
from pathlib import Path

import numpy as np
from arms_model import (
    Upright,
    all_sets,
    delta_at,
    jacobian_with_arms,
    k_fold,
    numerators,
    pencils,
    quartic,
    quartic_from_jacobian,
    steer_stiffness_quasi_static,
)

from whipple.model import Whipple

REPO = Path(__file__).resolve().parents[2]
SETS = ("benchmark", "Browser+Jason", "Rigid+Jason")
TURNS = ((0.03, 0.01), (0.10, 0.05), (0.16, 0.20), (0.25, 0.10))
ARMS = ((0, 0, 0), (3.0, 0, 0), (0, 2.0, 0), (4.0, 1.5, 0.2), (60.0, 2.0, 0.4))


def main():
    sets = all_sets()
    out = {}
    worst = worst_kf = 0.0
    for name in SETS:
        bike = Whipple(sets[name])
        for phi, delta in TURNS:
            pen, _ = pencils(bike, phi, delta)
            for plant in ("free", "constant"):
                A, Lpp = numerators(*pen[plant])
                for k, c, Ia in ARMS:
                    a = quartic(A, Lpp, k, c, Ia)
                    b = quartic_from_jacobian(
                        jacobian_with_arms(bike, phi, delta, plant, k, c, Ia)[0]
                    )
                    worst = max(worst, np.max(np.abs(a - b) / (1 + np.abs(b))))
                kf = k_fold(A, Lpp)
                kq = -steer_stiffness_quasi_static(bike, phi, delta, plant)
                worst_kf = max(worst_kf, abs(kf - kq) / (1 + abs(kf)))
    out["quartic_formula_vs_jacobian_max_rel"] = worst
    out["k_fold_vs_quasi_static_max_rel"] = worst_kf

    u = Upright(sets["benchmark"])
    k0, kappa_c, _ = u.meijaard_form()
    out["benchmark"] = dict(
        k0=u.k0,
        kappa_c=u.kappa_c,
        k0_meijaard=k0,
        kappa_c_meijaard=kappa_c,
        V_a3=u.V_a3,
        v_c=u.v_c,
        v_w=u.v_w(),
        k_star=u.k_star,
        c_star=u.c_star,
        corner_a1_a2=u.corner(),
    )
    rows = []
    for k, c in (
        (0, 0),
        (1, 0),
        (3, 0),
        (5, 0),
        (5.5, 0),
        (5.7, 0),
        (0, 4),
        (0, 14),
        (0, 14.5),
        (3.2, 0.72),
        (5.0, 0.9),
    ):
        rows.append(
            dict(
                k=k,
                c=c,
                closed_form=u.band(k, c),
                eigenvalues=u.band_numeric(k, c, Vs=np.linspace(0.3, 8, 7701)),
            )
        )
    out["upright_bands"] = rows

    # A damper can make a single turn stable (benchmark, outside the region joined to upright).
    bike = Whipple(sets["benchmark"])
    phi, V = 0.301764705882353, 2.1875
    d = delta_at(bike, phi, V, 0.7)
    pen, turn = pencils(bike, phi, d)
    A, Lpp = numerators(*pen["free"])
    roots = {f"{c:g}": np.roots(np.r_[1.0, quartic(A, Lpp, 0.0, c)]) for c in (0.0, 0.5)}
    out["damped_turn"] = dict(
        lean_deg=math.degrees(phi),
        V=turn.V,
        delta=d,
        T=turn.T,
        max_real={c: float(r.real.max()) for c, r in roots.items()},
    )
    path = REPO / "data" / "rider" / "arms_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
