"""Tyre-slip tongue: stable turns of the benchmark bicycle at constant steer torque on Magic Formula
tyres with the rear-wheel speed held (Appendix F, Fig. a).

For each speed the lean is stepped up from upright and the edges of the stable region are found
where the largest real part of the 8 eigenvalues changes sign: a weave Hopf edge (complex pair) or
a fold edge (real eigenvalue). The Bogdanov-Takens point solves det J = 0 and c_7 = 0 for the
characteristic polynomial, and the largest stable lean is the maximum of the fold curve.
Reference numbers of the paper: BT point at 10.43 deg and 2.54 m/s, largest stable lean 18.73 deg.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from whipple import plotstyle as ps
from whipple.output import options, save_json
from whipple.tyre import TyreBike

BT_GUESS = (np.radians(10.4), 2.54)  # lean (rad) and speed (m/s) near the BT point


class Column:
    """Steady turns at the speed V along the lean, with continuation in the lean."""

    def __init__(self, bike: TyreBike, V):
        self.bike, self.V = bike, V
        self.wr, self.x = bike.upright(V), np.zeros(7)

    def eigenvalues(self, phi):
        wr, x, _ = self.bike.turn_at_speed(phi, self.V, self.x, self.wr)
        self.wr, self.x = wr, x
        return self.bike.turn_eigenvalues(phi, wr, x)

    def margin(self, phi):
        return self.eigenvalues(phi).real.max()

    def march(self, phi):
        """Continue from upright to the lean phi."""
        for p in np.linspace(0.0, phi, 12)[1:]:
            self.eigenvalues(p)


def edges(bike, V, phis):
    """Leans (rad) and kinds of the edges of the stable region at speed V."""
    col = Column(bike, V)
    out, prev = [], None
    for phi in phis:
        lam = col.eigenvalues(phi)
        stable = lam.real.max() < 0
        if prev is not None and stable != prev[1]:
            state = (col.wr, col.x)
            col.wr, col.x = prev[2]
            root = brentq(col.margin, prev[0], phi, xtol=1e-9)
            unstable = lam if not stable else prev[3]
            w = abs(unstable[np.argmax(unstable.real)].imag)
            out.append((root, "hopf" if w > 1e-6 else "fold"))
            col.wr, col.x = state
        prev = (phi, stable, (col.wr, col.x), lam)
    return out


def bt_point(bike, guess):
    """Newton iteration on the two last coefficients of the characteristic polynomial."""
    col = Column(bike, guess[1])
    col.march(guess[0])

    def f(p):
        col.V = p[1]
        return np.real(np.poly(col.eigenvalues(p[0])))[-1:-3:-1]

    x = np.array(guess, float)
    scale = np.abs(f(x))
    for _ in range(30):
        h = (1e-6, 1e-5)
        J = np.column_stack([(f(x + np.eye(2)[k] * h[k]) - f(x - np.eye(2)[k] * h[k]))
                             / (2 * h[k] * scale) for k in range(2)])  # fmt: skip
        dx = -np.linalg.solve(J, f(x) / scale)
        x = x + min(1.0, 0.01 / np.abs(dx).max()) * dx
        if abs(dx[0]) < 1e-10 and abs(dx[1]) < 1e-9:
            break
    return x


def main():
    args = options(__doc__)
    bike = TyreBike()
    step = 1.0 if args.quick else 0.5
    speeds = np.arange(2.5, 12.01, step)
    phis = np.radians(np.arange(step, 25.01, step))
    pts = {"hopf": [], "fold": []}
    for V in speeds:
        for phi, kind in edges(bike, V, phis):
            pts[kind].append((V, np.degrees(phi)))

    phi_bt, V_bt = bt_point(bike, BT_GUESS)
    col = Column(bike, V_bt)
    col.march(phi_bt)
    turn = bike.turn_speed(phi_bt, col.wr, col.x)

    def fold_lean(V):
        found = edges(bike, V, np.radians(np.arange(8.0, 25.01, 1.0)))
        return max(p for p, k in found if k == "fold")

    top = minimize_scalar(lambda V: -fold_lean(V), bounds=(5.0, 7.5), method="bounded",
                          options=dict(xatol=1e-3))  # fmt: skip
    save_json("tyre_tongue", {
        "BT lean (deg)": np.degrees(phi_bt),
        "BT speed V (m/s)": V_bt,
        "BT steer torque (N m)": col.x[5],
        "BT lateral acceleration a_y/g": turn["ay"] / ps.G,
        "largest stable lean (deg)": np.degrees(-top.fun),
        "largest stable lean at V (m/s)": top.x,
        "edge points hopf (V, lean deg)": pts["hopf"],
        "edge points fold (V, lean deg)": pts["fold"],
    })  # fmt: skip

    fig, ax = ps.figure(ps.HALF, 6.0 * ps.CM)
    for kind, color in (("hopf", ps.HOPF), ("fold", ps.FOLD)):
        p = np.array(sorted(pts[kind] + [(V_bt, np.degrees(phi_bt))]))
        ax.plot(p[:, 0], p[:, 1], "o-", color=color, ms=2, lw=ps.LW_THIN)
    ps.bt_marker(ax, V_bt, np.degrees(phi_bt), filled=True)
    ax.set(xlim=(2, 12), ylim=(0, 20), xlabel=ps.XV, ylabel=ps.YLEAN)
    ps.label(ax, V_bt - 0.2, np.degrees(phi_bt) + 0.8, "BT", ha="right", va="bottom")
    ps.label(ax, 7, 3, "Hopf", color=ps.HOPF)
    ps.label(ax, 9, 14, "Fold", color=ps.FOLD)
    ps.save(fig, "tyre_tongue")


if __name__ == "__main__":
    main()
