import numpy as np
import pytest

from whipple import stability as st
from whipple.bicycles import BENCHMARK
from whipple.linear import Linear
from whipple.model import Whipple

# One state of the benchmark bicycle from Basu-Mandal et al. (2007), Table 1, in the axes of
# Meijaard et al. (2007): lean = pi/2 - psi, steer = -psi_f, speeds with the opposite sign.
BASU_STATE = (np.pi / 2 - 0.9501292851472, -0.2311385135743)
BASU_SPEEDS = np.array([-0.6068425835418, -0.4859824687093, -8.912989661489])
BASU_PITCH = np.pi - 3.1257073014894
# Second derivatives of (x, y, z, yaw, lean, pitch, steer, rear wheel, front wheel) in the axes of
# Basu-Mandal et al.
BASU_QDD = np.array([-0.5041626315047, -0.3449706619454, -1.4604528332980, 0.8353281706379,
                     -7.8555281128244, 0.1205543897884, -4.6198904039403, 1.8472554144217,
                     2.454807290455])  # fmt: skip
BASU_SIGNS = np.array([-1, 1, -1, -1, -1, -1, -1, -1, -1])


@pytest.fixture(scope="module")
def bike():
    return Whipple(BENCHMARK)


@pytest.mark.parametrize("V", [1.0, 4.0, 4.6, 7.0])
def test_upright_jacobian_agrees_with_linear_model(bike, V):
    z = np.array([0.0, 0.0, 0.0, 0.0, -V / bike.p["rR"]])
    J = bike.jacobian(z)
    lin = Linear(BENCHMARK)
    assert np.allclose(J[:4, :4], lin.state_matrix(V), atol=1e-10)


def test_upright_weave_and_capsize_speeds():
    lin = Linear(BENCHMARK)
    assert lin.weave_speed() == pytest.approx(4.292383, abs=1e-6)
    assert lin.capsize_speed() == pytest.approx(6.024262, abs=1e-6)


def test_basu_mandal_accelerations(bike):
    assert bike.pitch(*BASU_STATE) == pytest.approx(BASU_PITCH, abs=1e-12)
    qdd = bike.qdd(*BASU_STATE, BASU_SPEEDS) * BASU_SIGNS
    assert np.max(np.abs(qdd - BASU_QDD)) < 1e-10


def test_basu_mandal_hands_free_turn(bike):
    # First hands-free turn of Basu-Mandal's thesis, Tables 6.1 and 7.1.
    phi, delta, spin_rate = 1.9893886377 - np.pi / 2, -3.0755121969, 26.3580011755
    z = np.array([phi, delta, 0.0, 0.0, -spin_rate])
    assert np.max(np.abs(bike.field(z))) < 1e-8
    lam = np.linalg.eigvals(bike.jacobian(z))
    lam = np.sort_complex(np.delete(lam, np.argmin(np.abs(lam))))
    published = np.sort_complex([0.038127379, -21.152660576, -2.265960434 + 7.986013290j,
                                 -2.265960434 - 7.986013290j])  # fmt: skip
    assert np.max(np.abs(lam - published)) < 1e-8


@pytest.mark.parametrize("phi, delta", [(0.1, 0.05), (0.25, 0.3), (-0.4, -0.1), (0.6, 0.02)])
def test_capsize_condition_identity(bike, phi, delta):
    product = st.characteristic(bike, phi, delta)[3]
    assert st.capsize_formula(bike, phi, delta) == pytest.approx(product, rel=1e-9)


def test_tip_of_stable_region(bike):
    phi, delta = st.bogdanov_takens(bike, (0.1616, 0.2265))
    turn = bike.turn(phi, delta)
    assert np.degrees(turn.phi) == pytest.approx(9.257, abs=5e-4)
    assert turn.V == pytest.approx(2.7896, abs=5e-5)


def test_largest_stable_lean(bike):
    def fold(x):
        return st.characteristic(bike, *x)[3]

    x = st.boundary_extremum(bike, fold, lambda t: t.phi, (0.2177, 0.1126))
    turn = bike.turn(*x)
    assert np.degrees(turn.phi) == pytest.approx(12.47, abs=5e-3)
    assert turn.V == pytest.approx(4.536, abs=5e-4)
