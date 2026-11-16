"""models.K_cei and models.SN -- K(m) by the arithmetic-geometric mean and
sn(x|m) by the descending Landen transformation (A&S 16.4, 17.6), one formula
for every 0 <= |m| < 1 -- against closed forms.

Pure functions, no window.
"""
import math

import numpy as np
import pytest

import syncmoss.models as m5

EPS = np.finfo(float).eps


def test_k_against_closed_forms():
    assert m5.K_cei(0.0) == pytest.approx(np.pi / 2, abs=4 * EPS)
    # Legendre: K(1/2) = Gamma(1/4)^2 / (4 sqrt(pi))
    assert m5.K_cei(0.5) == pytest.approx(math.gamma(0.25) ** 2 / (4 * math.sqrt(np.pi)), rel=4 * EPS)
    assert m5.K_cei(-0.5) == m5.K_cei(0.5)
    # near 1 (A&S 17.3.26): K = L + k'^2/4 (L - 1) + O(k'^4 L), L = ln(4/k'), k'^2 = 1 - m
    for m in (1 - 1e-8, 1 - 1e-12):
        kc2 = 1 - m                                                                  # exact: what K_cei sees
        L = math.log(4 / math.sqrt(kc2))
        assert m5.K_cei(m) == pytest.approx(L + kc2 / 4 * (L - 1), rel=16 * EPS)
    assert m5.K_cei(1.0) == 99999                                                    # the period is infinite


@pytest.mark.parametrize('m', [0.0, 0.3, 0.9, 0.981, 0.999999, 1 - 1e-15])
def test_sn_at_a_quarter_and_a_half_of_the_quarter_period(m):
    K = m5.K_cei(m)
    sn = m5.SN(np.array([0.0, K / 2, K]), m)
    # sn(K/2|m) = 1/sqrt(1 + k'), k' = sqrt(1 - m)  (A&S 16.5)
    assert sn == pytest.approx([0.0, 1 / math.sqrt(1 + math.sqrt(1 - m)), 1.0], abs=8 * EPS)


def test_sn_at_the_two_ends_of_m():
    x = np.linspace(0.0, 3.0, 31)
    assert m5.SN(x, 0.0) == pytest.approx(np.sin(x), abs=4 * EPS)
    # near 1 (A&S 16.15.1): sn = tanh x + k'^2/4 (sinh x cosh x - x) / cosh^2 x + O(k'^4)
    kc2 = 1e-6
    expansion = np.tanh(x) + kc2 / 4 * (np.sinh(x) * np.cosh(x) - x) / np.cosh(x) ** 2
    assert np.max(np.abs(m5.SN(x, 1 - kc2) - expansion)) < 10 * kc2 ** 2
    assert np.max(np.abs(m5.SN(x, 1 - kc2) - np.tanh(x))) > 100 * kc2 ** 2             # the correction is there
    assert np.all(m5.SN(x, 1.0) == 1.0) and np.all(m5.SN(x, -0.3) == m5.SN(x, 0.3))


def test_sn_has_no_step_in_m():
    # until 2026-11 sn switched to tanh above m = 0.981: a step of 4e-3
    x = np.linspace(0.0, m5.K_cei(0.981), 101)
    assert np.max(np.abs(m5.SN(x, 0.981 + 1e-9) - m5.SN(x, 0.981))) < 1e-7
