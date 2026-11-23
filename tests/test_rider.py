"""Tests of Sect. 13, "Rider": steer spring and damper of the arms, and the lateral mass shift."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "rider"))
DATA = REPO / "data" / "rider"

from arms_model import (  # noqa: E402
    Upright,
    all_sets,
    jacobian_with_arms,
    k_fold,
    numerators,
    pencils,
    quartic,
    quartic_from_jacobian,
    steer_stiffness_quasi_static,
)
from shift_model import totals  # noqa: E402

from whipple.bicycles import BICYCLES  # noqa: E402
from whipple.model import Whipple  # noqa: E402


def stored(name):
    return json.loads((DATA / name).read_text())


def test_closing_stiffness_benchmark():
    u = Upright(BICYCLES["benchmark"])
    assert u.kappa_c == pytest.approx(0.19457, abs=1e-5)
    assert u.k0 == pytest.approx(7.061, abs=1e-3)
    assert u.k_star == pytest.approx(5.578, abs=1e-3)
    assert u.k_star == pytest.approx(u.kappa_c * (u.v_c**2 - u.V_a3**2), rel=1e-12)
    k0, kappa_c, _ = u.meijaard_form()
    assert (k0, kappa_c) == pytest.approx((u.k0, u.kappa_c), rel=1e-12)
    assert u.band() == pytest.approx((4.29, 6.02), abs=6e-3)


def test_damping_threshold_and_growth_rates():
    u = Upright(BICYCLES["benchmark"])
    assert u.c_star == pytest.approx(14.17, abs=5e-3)
    assert np.roots(np.r_[1.0, u.a(5.0, 10.0)]).real.max() == pytest.approx(0.85, abs=0.005)
    assert math.sqrt(-u.Kpp / u.Mpp) == pytest.approx(3.1, abs=0.05)


@pytest.mark.parametrize("plant", ["free", "constant"])
def test_capsize_condition_with_spring(plant):
    """Proposition prop:rider-fold(i): the fold stiffness is -dT/d(delta) on the loading path."""
    bike = Whipple(BICYCLES["benchmark"])
    phi, delta = 0.16, 0.20
    A, Lpp = numerators(*pencils(bike, phi, delta)[0][plant])
    kq = -steer_stiffness_quasi_static(bike, phi, delta, plant)
    assert k_fold(A, Lpp) == pytest.approx(kq, rel=1e-9)
    a = quartic(A, Lpp, 4.0, 1.5, 0.2)
    b = quartic_from_jacobian(jacobian_with_arms(bike, phi, delta, plant, 4.0, 1.5, 0.2)[0])
    assert a == pytest.approx(b, rel=1e-9)


def test_closing_stiffness_measured_sets():
    s = stored("arms_summary.json")
    S = s["summary"]["k_star_riders"]
    assert (S["min"], S["max"], S["median"]) == pytest.approx((0.98, 15.1, 5.2), abs=0.06)
    assert S["n_between_3p2_and_7p2"] == 18
    for name, p in all_sets().items():
        assert Upright(p).k_star == pytest.approx(s["sets"][name]["k_star"], rel=1e-10)
    assert s["summary"]["k_star_scan_max_abs"] < 0.0025


def test_largest_lean_with_spring():
    lean = stored("arms_summary.json")["sets"]["benchmark"]["free"]["lean_star"]
    assert [lean["0"], lean["0.4"], lean["0.8"]] == pytest.approx([12.47, 8.13, 4.04], abs=0.006)


def test_equivalent_torque_of_shift():
    p = BICYCLES["benchmark"]
    mT, h, SA = totals(p)
    assert p["g"] * SA * math.pi / 180 == pytest.approx(0.445, abs=5e-4)
    assert SA / (mT * h) * 75 * p["g"] * 0.01 == pytest.approx(0.236, abs=5e-4)


def test_hands_free_largest_lean():
    s = stored("shift_summary.json")["benchmark"]["free"]
    assert s["hands_free_deg"] == pytest.approx(13.59, abs=0.005)
    assert s["hands_free_V"] == pytest.approx(4.26, abs=0.005)
    assert s["torque_deg"] == pytest.approx(12.47, abs=0.005)
    assert s["largest_frame_lean_deg"] == pytest.approx(12.85, abs=0.005)
    c = stored("shift_summary.json")["benchmark"]["constant"]
    assert (c["hands_free_deg"], c["torque_deg"]) == pytest.approx((15.16, 13.84), abs=0.005)


def test_largest_hands_free_shift():
    assert stored("shift_simulations.json")["largest_shift"]["Delta_deg"] == pytest.approx(
        0.559, abs=5e-4
    )


def test_fixed_shift_grid():
    rows = {
        round(r["Delta"], 3): r["phiG"] for r in stored("shift_connected.json")["benchmark_free"]
    }
    got = [rows[d] for d in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
    assert got == pytest.approx([9.5, 11.0, 12.0, 14.0, 16.0, 18.5, 22.5])
