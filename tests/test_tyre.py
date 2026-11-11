import numpy as np
import pytest
from scipy.optimize import brentq

from whipple.tyre import TyreBike


@pytest.fixture(scope="module")
def bike():
    return TyreBike()


def test_blind_point_of_the_tyre_model(bike):
    # 6 m/s rear-centre speed, 30 deg lean, nominal Magic Formula tyre: steer 8.4716 deg, steer
    # torque -0.30910 N m, capsize +0.96758 1/s, weave -1.9537 +- 7.6635i 1/s.
    wr, x = -20.49278181654, np.zeros(7)
    for phi in np.radians(np.arange(1.0, 30.01, 1.0)):
        x, res = bike.solve_turn(phi, wr, x)
    assert res < 1e-10
    assert np.degrees(x[0]) == pytest.approx(8.4716, abs=1e-4)
    assert x[5] == pytest.approx(-0.30910, abs=1e-5)
    lam = bike.turn_eigenvalues(phi, wr, x)
    assert lam.real.max() == pytest.approx(0.96758, abs=1e-5)
    weave = lam[np.abs(lam.imag) == np.sort(np.abs(lam.imag))[-5]]
    assert np.abs(weave.imag).max() == pytest.approx(7.6635, abs=1e-4)
    assert weave.real.max() == pytest.approx(-1.9537, abs=1e-4)


def test_upright_weave_speed_with_tyres(bike):
    def margin(V):
        lam = bike.turn_eigenvalues(0.0, bike.upright(V), np.zeros(7))
        return lam[(np.abs(lam.imag) > 1e-6) & (np.abs(lam.imag) < 15)].real.max()

    # The tyres move the no-slip weave speed 4.2924 m/s up by 0.74 m/s.
    assert brentq(margin, 4.5, 5.5, xtol=1e-8) == pytest.approx(5.0354, abs=2e-4)
