"""Capsize condition in a turn, Eq. (10): the product of the four nonzero eigenvalues equals
-2 F_c,phi / (m_s det M_eff) times det d(T, H)/d(lean, steer), at random steady turns."""

import numpy as np

from whipple.bicycles import BICYCLES
from whipple.model import Whipple
from whipple.output import options, save_json
from whipple.stability import capsize_formula, characteristic


def main():
    args = options(__doc__)
    n = 10 if args.quick else 140
    rng = np.random.default_rng(2026)
    rows = []
    for name, p in BICYCLES.items():
        bike = Whipple(p)
        count = 0
        while count < n:
            phi, delta = rng.uniform(-0.8, 0.8), rng.uniform(-0.6, 0.6)
            turn = bike.turn(phi, delta)
            if turn is None or not 0.5 < turn.V < 15:
                continue
            lhs = characteristic(bike, phi, delta)[3]
            rhs = capsize_formula(bike, phi, delta)
            rows.append(
                dict(bicycle=name, lean=phi, steer=delta, V=turn.V, product=lhs, formula=rhs)
            )
            count += 1
    rel = [
        abs(r["product"] - r["formula"]) / max(abs(r["product"]), abs(r["formula"])) for r in rows
    ]
    signs = sum(np.sign(r["product"]) == np.sign(r["formula"]) for r in rows)
    save_json("capsize_condition", {
        "turns": len(rows),
        "largest relative difference": max(rel),
        "median relative difference": float(np.median(rel)),
        "same sign": int(signs),
        "turns tested": rows,
    })  # fmt: skip


if __name__ == "__main__":
    main()
