"""Capsize rules and the quartic to second order in the lean (paper Sects. 6 and 8)."""

import numpy as np
import pytest

from whipple import capsize_rule as cr
from whipple.bicycles import BENCHMARK, measured_sets, measured_sources
from whipple.linear import Linear
from whipple.quartic import Quartic, bracket


def test_tau1_is_the_linear_torque_coefficient():
    # eq. tau1: tau1 = -det K / K_phidelta
    for V in (2.0, 5.0, 7.0):
        K = Linear(BENCHMARK).K(V)
        tau1 = cr.coefficients(BENCHMARK, V)["tau1"]
        assert tau1 == pytest.approx(-np.linalg.det(K) / K[0, 1], rel=1e-9)


def test_capsize_speed_of_the_series():
    assert cr.series(BENCHMARK).v_c == pytest.approx(6.024262, abs=1e-6)


def test_second_order_rule_benchmark():
    # Paper Sect. 6.3: largest phi_c of the second-order rule 12.45 deg at 4.54 m/s.
    r = cr.largest_lean_rule(BENCHMARK, "free", order=2)
    assert r["phi_deg"] == pytest.approx(12.4526, abs=1e-3)
    assert r["V"] == pytest.approx(4.543, abs=2e-3)
    assert r["valid"]


def test_first_order_and_constant_speed_rules_benchmark():
    assert cr.largest_lean_rule(BENCHMARK, "free", order=1)["phi_deg"] == pytest.approx(
        12.6066, abs=1e-3
    )
    # Constant speed: phi_c = sqrt(-tau1 / (3 tau3)), eq. H1.
    assert cr.largest_lean_rule(BENCHMARK, "constant", order=1)["phi_deg"] == pytest.approx(
        13.9815, abs=1e-3
    )
    c = cr.coefficients(BENCHMARK, 4.7)
    assert cr.phi_c_constant(BENCHMARK, 4.7) == pytest.approx(np.sqrt(-c["tau1"] / (3 * c["tau3"])))


def test_two_thirds_rule_at_cubic_order():
    # eq. twothirds: the minimum of tau1 phi + tau3 phi^3 is (2/3) tau1 phi_c.
    c = cr.coefficients(BENCHMARK, 5.0)
    phi = np.sqrt(-c["tau1"] / (3 * c["tau3"]))
    T = c["tau1"] * phi + c["tau3"] * phi**3
    assert T == pytest.approx(2 / 3 * c["tau1"] * phi, rel=1e-12)


def test_plant_names_are_checked():
    with pytest.raises(ValueError):
        cr.largest_lean_rule(BENCHMARK, "held")


def test_tip_speed_formula_benchmark():
    # eq. Va3: V_a3 = 2.761 m/s for the benchmark bicycle.
    q = Quartic(BENCHMARK)
    assert q.v_a3() == pytest.approx(2.7608, abs=1e-4)
    assert q.a0(q.v_a3())[2] == pytest.approx(0.0, abs=1e-10)


def test_quartic_upright_coefficients_match_linear_model():
    q, lin = Quartic(BENCHMARK), Linear(BENCHMARK)
    for V in (1.0, 4.0, 6.5):
        assert np.allclose(np.poly(lin.eigenvalues(V))[1:].real, q.a0(V), atol=1e-9)


def test_bracket_identity():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    assert np.linalg.det(A + B) == pytest.approx(
        np.linalg.det(A) + bracket(A, B) + np.linalg.det(B)
    )


def test_measured_sets_load():
    rider, bare = measured_sets("rider"), measured_sets("riderless")
    assert len(rider) == 23
    assert set(rider["Crescendo+Jason"]) == set(BENCHMARK)
    assert "Silver" in bare
    assert all("source" in v for v in measured_sources("rider").values())


def test_constant_speed_plant_name_is_an_alias_of_held():
    from whipple import stability as st
    from whipple.model import Whipple

    bike = Whipple(BENCHMARK)
    a = st.characteristic(bike, 0.15, 0.1, "constant")
    assert np.array_equal(a, st.characteristic(bike, 0.15, 0.1, "held"))
    assert not np.allclose(a, st.characteristic(bike, 0.15, 0.1, "free"))
