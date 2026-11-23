"""Fast independent recomputations of main numbers of the paper (scripts/checks/independent.py)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from whipple.bicycles import BENCHMARK

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "checks" / "independent.py"
_SPEC = importlib.util.spec_from_file_location("independent", _PATH)
ind = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ind)


@pytest.fixture(scope="module")
def bike():
    return ind.Bike(BENCHMARK)


def test_upright_speeds_and_spring_constants():
    u = ind.upright_speeds(BENCHMARK)
    assert u["v_w"] == pytest.approx(4.292, abs=5e-4)
    assert u["v_c"] == pytest.approx(6.024, abs=5e-4)
    assert u["V_a3"] == pytest.approx(2.761, abs=5e-4)
    closed, elim = (
        ind.spring_constants_closed(BENCHMARK),
        ind.spring_constants_elimination(BENCHMARK),
    )
    assert closed["kappa_c"] == pytest.approx(elim["kappa_c"], rel=1e-12)
    assert closed["V_a3"] == pytest.approx(u["V_a3"], rel=1e-12)
    # Sect. 13.2: k_c = 7.061 - 0.1946 V^2 N m/rad, k* = 5.578 N m/rad.
    assert elim["kappa_c"] == pytest.approx(0.1945735, abs=1e-7)
    assert elim["k0"] == pytest.approx(7.061, abs=5e-4)
    kstar = elim["kappa_c"] * (u["v_c"] ** 2 - u["V_a3"] ** 2)
    assert kstar == pytest.approx(5.57837, abs=1e-5)
    assert ind.upright_coefficients(BENCHMARK, u["V_a3"], k=kstar)[3] == pytest.approx(0, abs=1e-9)


def test_bt_point_of_the_benchmark(bike):
    # Sect. 5.2 and 8.2: BT point at 9.26 deg and 2.790 m/s, T = -0.858 N m.
    (V, phi), a = ind.bt_point(bike, [2.79, 0.1616])
    assert V == pytest.approx(2.790, abs=5e-4)
    assert math.degrees(phi) == pytest.approx(9.257, abs=5e-4)
    assert bike.trim(V, phi)[1] == pytest.approx(-0.858, abs=5e-4)
    assert a[1] > 0


def test_largest_stable_lean_on_the_fold(bike):
    # Sect. 5.2: largest stable lean 12.47 deg at 4.54 m/s; the fold lean is stationary there.
    leans = [ind.fold_lean(bike, V, phi_max=0.25, n=6) for V in (4.50, 4.535, 4.57)]
    assert math.degrees(leans[1]) == pytest.approx(12.4725, abs=1e-3)
    assert leans[1] > max(leans[0], leans[2])


def test_theorem_1_and_constant_speed_fold(bike):
    for V, phi in ((3.5, 0.12), (6.5, 0.3)):
        f = ind.fold_identity(bike, V, phi)
        for plant in ("free", "constant"):
            jac, det = f[plant]
            assert jac == pytest.approx(det, rel=1e-9, abs=1e-9)


def test_pencil_feedback_and_power(bike):
    for plant in ("free", "constant"):
        coeff = ind.pencil_coefficients(*ind.pencil_matrices(bike, 4.0, 0.15, plant))
        assert np.allclose(coeff, bike.quartic(4.0, 0.15, plant), rtol=1e-10)
    p, family = ind.feedback_identity(bike, 4.0, 0.15, 2.0)
    assert np.allclose(p, family, rtol=1e-10)
    z = np.array([0.2, -0.1, 0.3, -0.5, -15.0])
    assert abs(ind.power_residual(bike, z, 0.7)) < 1e-10
