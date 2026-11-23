"""Tests of the energy results (Sect. 3, "Energy")."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from whipple.bicycles import BENCHMARK
from whipple.model import Whipple

DATA = Path(__file__).resolve().parent.parent / "data" / "energy"
BIKE = Whipple(BENCHMARK)


def turn(phi_deg, V):
    phi = math.radians(phi_deg)

    def res(d):
        return BIKE.p["rR"] * math.sqrt(BIKE.turn_spin_torque(phi, d)[0]) - V

    return BIKE.turn(phi, brentq(res, 0.03, 0.1, xtol=1e-15))


def test_first_integral_and_power_balance():
    # dE/dt = T ddelta/dt, so H = E - T delta is constant (E is about 2000 J).
    t = turn(8.0, 5.0)
    z0 = BIKE.turn_state(t) + np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    sol = solve_ivp(
        lambda _, z: BIKE.field(z, t.T),
        (0, 2.0),
        z0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
        t_eval=np.linspace(0, 2.0, 41),
    )
    E = np.array([BIKE.energy(z) for z in sol.y.T])
    assert np.ptp(E - t.T * sol.y[1]) < 1e-9
    assert np.ptp(E) > 1e-3  # the steer torque does work
    assert E[0] - BIKE.energy(BIKE.turn_state(t)) == pytest.approx(6.60, abs=0.005)


def test_no_drive_torque_at_a_steady_turn():
    # Q = M_3q qdd - f_3 with s constant is zero at a steady turn.
    t = turn(10.0, 5.0)
    z = BIKE.turn_state(t)
    _, Mr, f, _ = BIKE.parts(z[0], z[1], z[2:5])
    qdd = np.linalg.solve(Mr[:2, :2], f[:2] + np.array([0.0, t.T]))
    assert abs(Mr[2, :2] @ qdd - f[2]) < 1e-12
    assert np.abs(qdd).max() < 1e-12


def test_equivalent_mass():
    assert BIKE.m_tot == pytest.approx(97.6, abs=0.05)


def test_stored_energy_numbers():
    d = json.loads((DATA / "disturbed_motions.json").read_text())
    rec, cap = d["recovery"], d["capsize"]
    assert rec["free"]["kick_energy"] == pytest.approx(6.60, abs=0.005)
    assert rec["free"]["predicted_end"]["V"] == pytest.approx(5.0164, abs=5e-5)
    assert rec["free"]["predicted_end"]["lean_deg"] == pytest.approx(8.240, abs=5e-4)
    assert rec["free"]["H_drift_max"] < 1e-10
    row = cap["table"][-1]
    assert row["U_drop"] == pytest.approx(215.5, abs=0.05)
    assert row["share_Kf"] == pytest.approx(0.994, abs=5e-4)
    assert row["V_estimate"] == pytest.approx(5.02, abs=0.005)
    assert row["V_rear_contact"] == pytest.approx(6.03, abs=0.005)
    assert row["V_mass_centre"] == pytest.approx(4.84, abs=0.005)
    w = json.loads((DATA / "growing_weave.json").read_text())["kick_4p1"]["upright"]
    c = w["change_of_cycle_means"]
    assert [c["U"], c["Kl"], c["Kf"]] == pytest.approx([-3.65, 3.62, 0.03], abs=0.006)
