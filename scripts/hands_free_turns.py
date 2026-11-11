"""Hands-free steady turns of the benchmark bicycle from Basu-Mandal, Chatterjee and Papadopoulos
(2007): equation residuals, turn radii and eigenvalues (Appendix D, Table 4)."""

import numpy as np

from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.output import options, save_json

# Thesis Table 6.1: roll psi (pi/2 upright), steer, rear-wheel spin rate beta_r' (rad/s), radius of
# the rear-wheel centre (m).
TURNS = [
    (1.9893886377, -3.0755121969, 26.3580011755, 13.8724247186),
    (1.9178291654, 0.4049333918, 10.3899258905, 2.2588798195),
    (1.7670024274, 0.7254537952, 5.5494771350, 1.1408878065),
    (1.7183161276, 0.8549190153, 4.2289953550, 0.8939154494),
    (2.1950752979, 0.4266815552, 14.4337001146, 1.7525375246),
    (2.0419972895, -2.6133787369, 10.9563251310, 1.4016100055),
    (2.3535106155, -2.8688460258, 19.4180569764, 2.3503396652),
]
# Thesis Table 7.1: the four nonzero eigenvalues (1/s).
EIGENVALUES = [
    [0.038127379, -21.152660576, -2.265960434 + 7.986013290j, -2.265960434 - 7.986013290j],
    [1.989869132, -4.886076369, -2.744979704 + 5.459259375j, -2.744979704 - 5.459259375j],
    [3.091516610, -2.853827876, -2.485975489 + 5.783418042j, -2.485975489 - 5.783418042j],
    [3.393903081, -2.608053659, -2.342566567 + 5.945917170j, -2.342566567 - 5.945917170j],
    [-2.000953101, -7.982680274, 5.575539147 + 5.799303852j, 5.575539147 - 5.799303852j],
    [-8.659556236, -0.795208976, -0.118995944 + 3.110982262j, -0.118995944 - 3.110982262j],
    [-13.209338580, -0.467653580, -0.075503592 + 7.402547429j, -0.075503592 - 7.402547429j],
]


def compare(bike, row, published):
    psi, delta, spin_rate, radius = row
    phi = psi - np.pi / 2
    z = np.array([phi, delta, 0.0, 0.0, -spin_rate])
    lam = np.linalg.eigvals(bike.jacobian(z, 0.0))
    lam = np.sort_complex(np.delete(lam, np.argmin(np.abs(lam))))
    turn = bike.turn(phi, delta)
    return {
        "lean (rad)": phi,
        "steer (rad)": delta,
        "largest residual": float(np.max(np.abs(bike.field(z, 0.0)))),
        "rear-centre radius (m)": turn.R - bike.p["rR"] * abs(np.sin(phi)),
        "published radius (m)": radius,
        "eigenvalues (1/s)": list(lam),
        "largest eigenvalue difference (1/s)": float(
            np.max(np.abs(lam - np.sort_complex(published)))
        ),
    }


def main():
    options(__doc__)
    bike = Whipple(BENCHMARK)
    save_json("hands_free_turns", [compare(bike, r, e) for r, e in zip(TURNS, EIGENVALUES)])


if __name__ == "__main__":
    main()
