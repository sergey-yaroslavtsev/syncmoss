"""dual_numbers: a calculation written with numpy returns, on Duals, its exact
derivatives in the fit parameters. Every case is one function run twice -- on
plain numbers, for central differences, and on seeded Duals -- so the
reference is independent of the chain rule under test.

Pure numpy: no window, no pool, no numba.
"""
import pickle
import warnings

import numpy as np
import pytest

import syncmoss.dual_numbers as dn
from syncmoss.expression_eval import eval_expr

X = np.linspace(-1.4, 2.6, 9)             # no point at 0, nor where a case has a kink
P = [1.3, 0.7, 2.1, 0.4]
FREE = [0, 1, 2]                          # p[3] is not fitted: it stays a plain number

# A central difference with the step eps^(1/3) (which balances its rounding
# and truncation errors) is good to eps^(2/3) of the function's scale.
STEP = np.finfo(float).eps ** (1 / 3)
ACCURACY = 1e3 * np.finfo(float).eps ** (2 / 3)


def _differences(f, p, free):
    rows = []
    for k in free:
        h = STEP * max(abs(p[k]), 1.0)
        up, down = list(p), list(p)
        up[k] += h
        down[k] -= h
        rows.append((np.asarray(f(up)) - np.asarray(f(down))) / (2 * h))
    return np.array(rows)


def _assert_jacobian(f, p=P, free=FREE):
    """f on Duals has f's value and the Jacobian the differences give; returns
    the Jacobian."""
    y = f(dn.seed(p, free))
    plain = np.asarray(f(list(p)))
    np.testing.assert_array_equal(dn.value(y), plain)
    J = dn.jacobian(y, free)
    reference = _differences(f, p, free)
    assert J.shape == reference.shape
    scale = max(np.max(np.abs(reference)), np.max(np.abs(plain)), 1.0)
    assert np.max(np.abs(J - reference)) < ACCURACY * scale
    return J


def _matrices(p):
    """Two stacks of 2x2 complex matrices over X, multiplied, transposed,
    conjugated and read out as the models read a layer out."""
    c, s = np.cos(p[0]), np.sin(p[0])
    M = dn.array([[c * c, c * s], [c * s, s * s]], dtype=complex)
    A = (np.exp(-p[1] * X ** 2) + 1j * X * p[2])[:, None, None] * M[None, :, :] + (1.0 - p[1]) * np.eye(2)
    B = np.exp(1j * p[2] * X)[:, None, None] * np.array([[0.0, 1.0], [-1.0, 0.0]])[None, :, :] + M
    T = np.matmul(A, B)
    gram = np.matmul(np.conjugate(np.transpose(T, (0, 2, 1))), T)
    return 0.7 * np.real(gram[:, 0, 0]) + 0.3 * np.real(gram[:, 1, 1]) + np.imag(T[:, 0, 1])


def _filled(p):
    """An array filled part by part, as _expm_neg fills its result."""
    out = np.empty_like(p[0] * X)
    out[:4] = np.exp(p[1] * X[:4])
    out[4:] = p[2] * X[4:] ** 2 + p[3] * p[0]
    out[0] = 5.0
    return out


CASES = {
    'arithmetic': lambda p: p[0] + X * p[1] - p[2] / (X + 3) + 2 / p[0] - (-p[1]) * p[2] + p[3] * p[0] + (+p[1]),
    'powers': lambda p: p[0] ** 2 + p[1] ** 0.5 + 2.0 ** p[2] + p[0] ** p[1] + (X + 2) ** p[2] + np.power(p[1], 3)
                        + np.square(p[2] - X),
    'exp, log, sqrt': lambda p: np.exp(-X ** 2 / p[0]) + np.log(p[1] + X ** 2) + np.log10(p[2]) + np.sqrt(p[0] + X ** 2),
    'trigonometric': lambda p: np.sin(p[0] * X) + np.cos(p[1]) * np.tan(0.3 * p[2]) + np.arcsin(0.1 * p[0] * X / 3)
                               + np.arccos(0.2 * p[1]) + np.arctan(p[2] * X),
    'hyperbolic': lambda p: np.sinh(p[0] * X / 3) + np.cosh(p[1]) * np.tanh(p[2] * X) + np.arcsinh(p[0] * X)
                            + np.arccosh(1 + p[1]) + np.arctanh(0.1 * p[2] * X / 3),
    'abs, max, min, hypot': lambda p: np.abs(X * p[0]) + abs(-p[1]) + np.maximum(p[0] * X, p[1]) + np.minimum(p[2], X ** 2)
                                      + np.hypot(p[0], X * p[1]) + max(0.0, p[2]) + min(p[0], 9.0),
    'complex': lambda p: np.abs(np.exp(1j * p[0] * X) * (p[1] + 1j * p[2])) + np.real(np.exp(1j * p[0] * X) * p[1])
                         + np.imag(np.conjugate(p[2] * np.exp(1j * X * p[1]))) + (p[0] * np.exp(1j * X)).real
                         + (p[1] * np.exp(1j * X)).imag,
    'indexing and broadcasting': lambda p: ((p[0] * X)[::2][:, None] * (p[1] * X[:3])[None, :] + (p[2] * X)[-1])[1:, ::-1]
                                           + (p[0] * X[:3].reshape(1, 3)).transpose()[:, 0],
    'sums': lambda p: np.sum(np.outer(X, X[:3]) * p[0] + p[1], axis=0) + np.mean(p[0] * X ** 2 + p[1] * X)
                      + np.std(p[2] * X) + np.var(p[1] * X + p[0]) + (p[0] * np.outer(X, X[:3])).sum(axis=-2)
                      + np.sum(p[2] * np.outer(X, X[:3]) ** 2, axis=0, keepdims=True)[0]
                      + (p[1] * np.outer(X, X[:3])).reshape(3, 9).mean(axis=1) + np.ravel(p[0] * np.outer(X[:3], X[:1])),
    'where': lambda p: np.where(X > 0, p[0] * X, p[1] * X ** 2) + np.where(X > 1, 2.0, p[2]) + np.where(X < 0, p[0], 1.0),
    'a distribution axis': lambda p: np.exp(-np.linspace(p[0], p[0] + p[1], 7) ** 2) * np.linspace(0.0, p[2], 7),
    'matrices': _matrices,
    'filling an array': _filled,
}


@pytest.mark.parametrize('name', CASES)
def test_the_derivatives_are_those_of_the_differences(name):
    J = _assert_jacobian(CASES[name])
    assert np.all(np.any(J != 0, axis=1))           # every fitted parameter is in the case


def test_only_the_parameters_a_value_depends_on_are_carried():
    q = dn.seed(P, FREE)
    assert not isinstance(q[3], dn.Dual)            # not fitted: a plain number
    assert (q[0] * X).idx == (0,)
    y = q[2] * np.exp(q[0] * X)                     # joined: the union, sorted
    assert y.idx == (0, 2) and y.d.shape == (len(X), 2)
    J = dn.jacobian(y, FREE)
    assert not np.any(J[1]) and np.all(J[0] != 0) and np.all(J[2] != 0)
    assert not np.any(dn.jacobian(X * 2.0, FREE))   # a plain value: all zeros


def test_links_and_expressions_are_plain_arithmetic_on_the_seeded_list():
    def linked(p):
        q = list(p)
        q[3] = 0.5 * q[0]                                     # a link
        q[1] = eval_expr("sqrt(p[0]) * exp(-p[2] / 3)", q)    # an Expression, read by the model below
        return q[3] * X + q[1] * X ** 2 + q[2]
    J = _assert_jacobian(linked, free=[0, 2])
    assert np.all(J != 0)


@pytest.mark.parametrize('text', [
    "sqrt(p[1]) * exp(-X**2 / 2 / p[0]**2) + abs(p[2]) * power(X, 2)",
    "mean(X * p[0]) + std(p[1] * X) + var(X) * p[2] + log(p[0]) * log10(p[1])",
    "sin(p[0] * X) + cos(p[1]) + tan(p[2] / 9) + arctan(X * p[0]) + sinh(p[1]) + cosh(p[2] / 9) + tanh(X * p[1])",
    "arcsin(p[0] / 9) + arccos(p[1] / 9) + arcsinh(p[2] * X) + arccosh(1 + p[0]) + arctanh(p[1] / 9) + pi * e * p[2]",
])
def test_a_user_text_is_differentiated_as_expression_eval_evaluates_it(text):
    # eval_expr itself, unchanged: its numpy functions take the Duals
    J = _assert_jacobian(lambda p: eval_expr(text, p, X) + 0 * X)
    assert np.all(np.any(J != 0, axis=1))


def test_the_functions_without_a_derivative_return_plain_numbers():
    q = dn.seed(P, FREE)
    for text in ("floor(p[0] * X)", "ceil(p[0] * X)", "round(p[0] * X)", "sign(p[0] * X)", "round(p[0] * X, 1)"):
        out = eval_expr(text, q, X)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, eval_expr(text, P, X))


def test_comparisons_are_those_of_the_values():
    q = dn.seed(P, FREE)
    y = q[0] * X
    for got, expected in ((y < 1, P[0] * X < 1), (y >= 1, P[0] * X >= 1), (y == y, np.full(len(X), True)),
                          (np.isfinite(y), np.full(len(X), True)), (X > y, X > P[0] * X), (y != 0.0, X != 0.0)):
        assert isinstance(got, np.ndarray) and got.dtype == bool
        np.testing.assert_array_equal(got, expected)
    assert q[0] > 1 and not q[1] > 1 and max(0.0, q[0]) is q[0] and len(y) == len(X)


def test_abs_at_zero_has_the_derivative_of_the_forward_difference():
    # |h d|/h = |d| along each parameter: the column of a parameter that
    # starts at 0 is not zero, whatever the sign of what multiplies it
    q = dn.seed([0.0, 0.0], [0, 1])
    np.testing.assert_array_equal(dn.jacobian(abs(q[0]), [0, 1]), [1.0, 0.0])
    assert X[0] < 0 < X[-1]
    np.testing.assert_array_equal(dn.jacobian(np.abs(q[1] * X), [0, 1]), [0 * X, np.abs(X)])
    np.testing.assert_array_equal(dn.jacobian(np.abs(q[1] * X * (3 + 4j)), [0, 1]), [0 * X, 5 * np.abs(X)])
    h = 1e-9
    np.testing.assert_allclose(np.abs(h * X) / h, np.abs(X), rtol=1e-12)


def test_sqrt_and_a_fractional_power_at_zero_give_zero_not_inf():
    q = dn.seed([0.0, 0.0], [0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter('error')              # no divide-by-zero warning either
        for y in (np.sqrt(q[0] ** 2 + q[1] ** 2), (q[0] ** 2 + q[1] ** 2) ** (1 / 2), np.sqrt(q[0]), q[0] ** 0.3,
                  np.std(q[0] * np.ones(3))):
            assert dn.value(y) == 0
            np.testing.assert_array_equal(dn.jacobian(y, [0, 1]), [0.0, 0.0])


def test_hypot_keeps_the_derivative_at_zero_zero():
    # sqrt(a^2 + b^2) of two widths that both start at 0: each parameter's
    # column is its own one-sided derivative, what its forward difference gives
    q = dn.seed([0.0, 0.0, 3.0], [0, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = np.hypot(2.0 * q[0], 0.5 * q[1] + 0.0 * q[2])
    np.testing.assert_array_equal(dn.jacobian(y, [0, 1, 2]), [2.0, 0.5, 0.0])
    h = 1e-9
    assert np.hypot(2.0 * h, 0.0) / h == pytest.approx(2.0) and np.hypot(0.0, 0.5 * h) / h == pytest.approx(0.5)
    # away from (0, 0), and where only one of the two is 0: the ordinary derivative
    _assert_jacobian(lambda p: np.hypot(p[0] * X, p[1]) + np.hypot(0.0 * p[2], p[2]))


@pytest.mark.parametrize('lose', [
    lambda y: float(y[0]),
    lambda y: complex(y[0]),
    lambda y: np.asarray(y),
    lambda y: np.array([y, y]),
    lambda y: np.zeros(len(X)).__setitem__(slice(None), y),     # stored in a plain array
    lambda y: np.add(np.zeros(len(X)), y, out=np.zeros(len(X))),
    lambda y: np.cumsum(y),                                     # a numpy function that is not in the tables
    lambda y: np.arctan2(y, 1.0),
    lambda y: np.interp(0.5, X, y),
])
def test_a_dual_is_never_turned_silently_into_a_plain_number(lose):
    y = dn.seed(P, FREE)[0] * X
    with pytest.raises(TypeError):
        lose(y)


def test_assigning_into_a_dual_leaves_what_it_was_made_from_untouched():
    q = dn.seed(P, FREE)
    y = q[0] * X
    same = y + 0.0
    part = y.real                                   # shares y's arrays
    part[0] = q[1]
    same[1] = 7.0
    np.testing.assert_array_equal(y.v, P[0] * X)
    np.testing.assert_array_equal(dn.jacobian(y, FREE), [X, 0 * X, 0 * X])
    assert part.idx == (0, 1) and part.v[0] == P[1] and same.v[1] == 7.0
    np.testing.assert_array_equal(dn.jacobian(part, FREE)[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(dn.jacobian(same, FREE)[:, 1], [0.0, 0.0, 0.0])


def test_a_dual_goes_to_a_pool_worker_and_back():
    y = np.exp(1j * dn.seed(P, FREE)[0] * X) * dn.seed(P, FREE)[2]
    back = pickle.loads(pickle.dumps(y))
    assert back.idx == y.idx
    np.testing.assert_array_equal(back.v, y.v)
    np.testing.assert_array_equal(back.d, y.d)


def test_the_derivatives_must_have_the_shape_of_the_value():
    with pytest.raises(ValueError):
        dn.Dual(np.zeros(3), np.zeros((3, 2)), (0,))
