"""Accuracy of the Hopf curve from the coefficients of the quartic (Sect. 8.2)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq, newton

from whipple import stability as st
from whipple.bicycles import BENCHMARK
from whipple.linear import Linear
from whipple.model import Whipple
from whipple.quartic import Quartic

DATA = Path(__file__).resolve().parent.parent / "data" / "weave" / "hopf_accuracy.json"


@pytest.fixture(scope="module")
def summary():
    return json.loads(DATA.read_text())["summary"]["free"]


def test_order2_accuracy(summary):
    """All four coefficients to order phi^2: largest lean error 0.77 deg (riderless Gyro)."""
    err, name = summary["order2_max_13_sets"]
    assert name == "Gyro" and err == pytest.approx(0.77, abs=0.01)


def test_hybrid_accuracy(summary):
    """Exact a4: 0.115 deg on 13 sets and the tip within about 1 % in speed.

    With Silver and Yellowrev and the tip point included it is 0.28 deg (not in the paper).
    """
    assert summary["hybrid_max_13_sets"][0] == pytest.approx(0.115, abs=0.003)
    assert summary["hybrid_max_with_Silver_Yellowrev"][0] == pytest.approx(0.28, abs=0.01)
    assert summary["hybrid_tip_speed_error_max_13_sets"] < 0.011


def test_eigenvalue_series_errors():
    d = json.loads(DATA.read_text())["summary"]
    lo = min(d[p]["eigen_error_range_13_sets"][0] for p in d)
    hi = max(d[p]["eigen_error_range_13_sets"][1] for p in d)
    assert hi == pytest.approx(7.4, abs=0.05)
    assert lo < 2.0  # the riderless Browser (zero-Hopf tip); the other sets are above 2 deg


def test_hybrid_benchmark_recomputed():
    """Benchmark, free rolling, at 3.36 m/s: exact Hopf lean 10.00 deg, hybrid within 0.05 deg."""
    bike, q, V = Whipple(BENCHMARK), Quartic(BENCHMARK), 3.36
    r = Linear(BENCHMARK).steer_ratio(V)
    cache = {}

    def delta(phi):
        if phi not in cache:
            g = r * phi

            def res(d):
                return bike.p["rR"] * math.sqrt(bike.turn_spin_torque(phi, d)[0]) - V

            cache[phi] = newton(res, g, x1=g * (1 + 1e-4), tol=1e-13, maxiter=100)
        return cache[phi]

    def exact(phi):
        return st.hurwitz(st.characteristic(bike, phi, delta(phi)))[0]

    def hybrid(phi):
        a = q.a(V, phi)
        a4 = st.capsize_formula(bike, phi, delta(phi))
        return a[0] * a[1] * a[2] - a[2] ** 2 - a[0] ** 2 * a4

    lo, hi = np.radians(9.0), np.radians(11.0)
    phi_e = math.degrees(brentq(exact, lo, hi, xtol=1e-9))
    phi_h = math.degrees(brentq(hybrid, lo, hi, xtol=1e-9))
    assert phi_e == pytest.approx(10.00, abs=0.01)
    assert 0 < phi_h - phi_e < 0.05
