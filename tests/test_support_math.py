"""Unit tests for support_math: numerical derivatives and error propagation.

These back the intensity-percentage and expression error columns shown in the
results table. They are pure numpy, so no GUI is required.
"""
import numpy as np
import pytest

from syncmoss.support_math import (
    calculate_partial_derivative_numerical,
    calculate_expression_error,
    calculate_intensity_percentage_error,
)


def test_partial_derivative_of_quadratic():
    # f(x) = x0^2 + 3*x1 ; df/dx0 = 2*x0 = 4 at x0=2 ; df/dx1 = 3
    f = lambda x: x[0] ** 2 + 3 * x[1]
    d0 = calculate_partial_derivative_numerical(f, [2.0, 5.0], 0, h=1e-6)
    d1 = calculate_partial_derivative_numerical(f, [2.0, 5.0], 1, h=1e-6)
    assert d0 == pytest.approx(4.0, abs=1e-3)
    assert d1 == pytest.approx(3.0, abs=1e-3)


def test_expression_error_sum_of_two_free_params():
    # f = p0 + p1, independent params -> var = var0 + var1
    params = np.array([2.0, 3.0])
    errors = np.array([0.1, 0.2])          # no NaN -> both free
    cov = np.diag([0.01, 0.04])            # variance 0.01 and 0.04
    err = calculate_expression_error("p[0]+p[1]", params, errors, cov)
    assert err == pytest.approx(np.sqrt(0.05), rel=1e-4)


def test_expression_error_scales_with_coefficient():
    # f = 2*p0 -> sigma_f = 2 * sigma_p0
    params = np.array([5.0])
    errors = np.array([0.1])
    cov = np.array([[0.01]])
    err = calculate_expression_error("2*p[0]", params, errors, cov)
    assert err == pytest.approx(0.2, rel=1e-4)


def test_expression_error_ignores_fixed_parameters():
    # p0 fixed (error NaN); covariance matrix only contains the free param p1.
    params = np.array([2.0, 3.0])
    errors = np.array([np.nan, 0.2])
    cov = np.array([[0.04]])               # 1x1: only p1 is free
    err = calculate_expression_error("p[0]+p[1]", params, errors, cov)
    assert err == pytest.approx(0.2, rel=1e-4)


def test_expression_error_all_fixed_is_zero():
    params = np.array([2.0, 3.0])
    errors = np.array([np.nan, np.nan])
    cov = np.zeros((0, 0))
    assert calculate_expression_error("p[0]+p[1]", params, errors, cov) == 0.0


def test_intensity_percentage_basic_split():
    # Two components with T=10 and T=30 -> 25% and 75%.
    t_values = np.zeros(6)
    t_values[2] = 10.0
    t_values[5] = 30.0
    t_errors = np.zeros(6)                 # all free, zero error
    cov = np.zeros((6, 6))
    intensities, intensity_errors = calculate_intensity_percentage_error(
        t_values, t_errors, cov, [2, 5]
    )
    assert intensities == pytest.approx([25.0, 75.0])
    assert intensity_errors == pytest.approx([0.0, 0.0])


def test_intensity_percentage_empty_indices():
    intensities, errors = calculate_intensity_percentage_error(
        np.array([1.0]), np.array([0.1]), np.zeros((1, 1)), []
    )
    assert intensities.size == 0
    assert errors.size == 0


def test_intensity_percentage_zero_total_returns_zeros():
    t_values = np.zeros(4)                  # all T == 0 -> sum == 0
    intensities, errors = calculate_intensity_percentage_error(
        t_values, np.zeros(4), np.zeros((4, 4)), [1, 2]
    )
    assert np.all(intensities == 0)
    assert np.all(errors == 0)


# --- the error propagation, computed each derivative once ---------------------
# The double loop over the parameter pairs used to recompute BOTH partial
# derivatives for every pair: ~4n^3 evaluations of an n-term expression for the
# % errors of n components of one spectrum -- 1.8 s for 20 components, minutes
# for 60. Every derivative is now computed once; the numbers must not change.

def _error_as_it_was(expr_str, parameters, errors, covariance_matrix, fixed_params=None):
    """calculate_expression_error before each derivative was computed once."""
    import re
    param_indices = sorted({int(m.group(1)) for m in re.finditer(r'p\[(\d+)\]', expr_str)})
    if not param_indices:
        return 0.0
    if fixed_params is None:
        fixed_params = np.array([i for i in range(len(errors)) if np.isnan(errors[i])], dtype=int)
    variable_params = [i for i in range(len(parameters)) if i not in fixed_params]
    param_to_cov_idx = {param: cov_idx for cov_idx, param in enumerate(variable_params)}
    variable_param_indices = [idx for idx in param_indices if idx not in fixed_params]
    if not variable_param_indices:
        return 0.0

    def expr_func(p):
        return eval(expr_str)

    variance = 0.0
    for i in variable_param_indices:
        for j in variable_param_indices:
            cov_i = param_to_cov_idx.get(i)
            cov_j = param_to_cov_idx.get(j)
            if cov_i is None or cov_j is None:
                continue
            if cov_i >= covariance_matrix.shape[0] or cov_j >= covariance_matrix.shape[1]:
                continue
            df_di = calculate_partial_derivative_numerical(expr_func, parameters, i)
            df_dj = calculate_partial_derivative_numerical(expr_func, parameters, j)
            variance += df_di * df_dj * covariance_matrix[cov_i, cov_j]
    return np.sqrt(abs(variance))


def _correlated_case(seed, n, fixed_share):
    rng = np.random.default_rng(seed)
    params = rng.uniform(0.1, 5.0, n)
    errors = rng.uniform(0.01, 0.2, n)
    errors[rng.random(n) < fixed_share] = np.nan
    n_free = int(np.sum(~np.isnan(errors)))
    root = rng.normal(size=(n_free, n_free))
    return params, errors, root @ root.T * 1e-3        # a correlated covariance


@pytest.mark.parametrize("seed, fixed_share", [(1, 0.0), (2, 0.3), (3, 0.7)])
@pytest.mark.parametrize("fixed_given", [False, True])
def test_expression_error_is_unchanged(seed, fixed_share, fixed_given):
    params, errors, cov = _correlated_case(seed, 24, fixed_share)
    fixed = [i for i in range(len(errors)) if np.isnan(errors[i])] if fixed_given else None
    for expr in ("p[3]*2+p[7]", "100*p[2]/(p[2]+p[5]+p[11]+p[23])",
                 "sqrt(p[1]**2+p[4]**2)".replace("sqrt", "np.sqrt"), "p[0]"):
        assert calculate_expression_error(expr, params, errors, cov, fixed) == \
            _error_as_it_was(expr, params, errors, cov, fixed), expr


def test_intensity_errors_are_unchanged():
    params, errors, cov = _correlated_case(4, 8 + 20, 0.3)
    t_indices = list(range(8, 28))                       # 20 components, one spectrum
    intensities, intensity_errors = calculate_intensity_percentage_error(
        params, errors, cov, t_indices)
    assert intensities.sum() == pytest.approx(100.0)
    sum_expr = '+'.join(f'p[{i}]' for i in t_indices)
    for k in (0, 13, 19):
        assert intensity_errors[k] == _error_as_it_was(
            f'100*p[{t_indices[k]}]/({sum_expr})', params, errors, cov)


def test_intensity_errors_of_many_components_are_quick():
    import time
    params, errors, cov = _correlated_case(5, 8 + 60, 0.0)
    start = time.perf_counter()
    calculate_intensity_percentage_error(params, errors, cov, list(range(8, 68)))
    assert time.perf_counter() - start < 10.0             # ~1 s; it was minutes
