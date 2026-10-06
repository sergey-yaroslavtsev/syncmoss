"""minimi_hi stops a parameter exactly on the bound the fit pushes it against.

y = a*x + b with the true slope outside a's bounds: the best fit has a on that
bound and b at its best value for it. A step that crossed an UPPER bound used to
put the parameter on its LOWER one -- the side was read with the index left over
from the loop before (the last parameter) instead of the parameter that hit --
so the step was refused, and the fit crept towards the bound and stopped short:
a = 0.99988 and b = 91.29 instead of 1 and 105.5. Both sides are checked, with
one and with two finite bounds, from two starts (b far off, and b already at its
best for the start of a, where moving a alone makes chi2 worse); and a fit that
does not reach its bounds.
"""
import numpy as np
import pytest

from syncmoss import minimi_lib as mi

X = np.linspace(1.0, 10.0, 50)
INF = np.inf

# name: (true slope, (lower, upper) bound of a, best a, best b)
CASES = {
    'upper bound': (2.0, (-INF, 1.0), 1.0, 105.5),
    'upper of two bounds': (2.0, (-0.5, 1.0), 1.0, 105.5),
    'lower bound': (-2.0, (-1.0, INF), -1.0, 94.5),
    'lower of two bounds': (-2.0, (-1.0, 0.5), -1.0, 94.5),
    'bounds not reached': (2.0, (-5.0, 5.0), 2.0, 100.0),
}


def _line(x, p):
    return p[0] * x + p[1]


@pytest.mark.parametrize('b_start', ['b far off', 'b adjusted'])
@pytest.mark.parametrize('case', list(CASES))
def test_a_parameter_ends_on_the_bound_it_is_pushed_against(case, b_start):
    slope, (low, high), best_a, best_b = CASES[case]
    y = slope * X + 100.0
    b0 = 90.0 if b_start == 'b far off' else 100.0 + slope * np.mean(X)   # a starts at 0
    bounds = np.array([[low, -INF], [high, INF]])

    p, errors, chi2, covariance = mi.minimi_hi(_line, X, y, np.array([0.0, b0]), bounds=bounds)

    assert p[0] == pytest.approx(best_a, abs=1e-9)
    assert p[1] == pytest.approx(best_b, abs=1e-6)


def _parabola(x, p):
    return p[0] * x + p[1] + p[2] * x ** 2


def test_a_step_crossing_the_lower_of_two_bounds_is_shortened_not_frozen():
    """With both bounds of a finite, a step crossing the lower one got the
    fraction 0 -- the upper-bound line overwrote the right one -- so the whole
    step was frozen and a alone thrown onto its bound. This fit (one of 20 in
    300 random ones) then stopped at a = -0.994, chi2 10 % above the best fit
    with a on its bound."""
    design = np.column_stack([X, np.ones_like(X), X ** 2])
    y = design @ np.array([-2.086, 114.854, -0.014])
    bounds = np.array([[-1.0, -INF, -INF], [0.5, INF, INF]])

    p, errors, chi2, covariance = mi.minimi_hi(_parabola, X, y, np.array([-0.122, 73.555, -0.022]),
                                               bounds=bounds)

    rest, *_ = np.linalg.lstsq(design[:, 1:], y + X, rcond=None)     # best b, c with a = -1
    assert p[0] == pytest.approx(-1.0, abs=1e-9)
    assert p[1:] == pytest.approx(rest, rel=1e-6)
