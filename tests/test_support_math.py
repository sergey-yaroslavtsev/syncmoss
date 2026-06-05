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
