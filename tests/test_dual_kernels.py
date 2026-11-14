"""dual_kernels: the twins of the njit line-shape kernels, held to the kernels
of models.py -- the value to rounding, the derivatives against the kernel's
own central differences -- and the models' matrix code (_expm_neg,
_polarised_readout, the texture blend), which takes Duals as it is written.

numba kernels, no window, no pool.
"""
import numpy as np
import pytest

import syncmoss.dual_kernels as dk
import syncmoss.dual_numbers as dn
import syncmoss.models as m5

S = np.linspace(-12.0, 12.0, 481)
# (gL, gG): Lorentzian- and Gaussian-dominated, the natural width alone
WIDTHS = [(0.097, 0.15), (0.3, 0.02), (0.02, 0.6), (0.097, 1e-3)]

EPS = np.finfo(float).eps
STEP = EPS ** (1 / 3)                 # of a central difference: good to eps^(2/3) of the function's scale
ACCURACY = 1e3 * EPS ** (2 / 3)


def _kernel(gL, gG, s):
    return m5.Voigt_complex(gL, gG, s, 1.0, 1j)


@pytest.mark.parametrize('gL, gG', WIDTHS + [(0.097, 0.0)])
def test_the_twin_has_the_value_of_the_kernel(gL, gG):
    A, D = dk._voigt_parts(gL, gG, S, True)
    reference = _kernel(gL, gG, S)
    peak = np.max(np.abs(reference))
    assert np.max(np.abs(A[0] + 1j * D[0] - reference)) < 100 * EPS * peak
    assert np.max(np.abs(A[0] - m5.Voigt(gL, gG, S))) < 100 * EPS * peak
    assert not np.any(dk._voigt_parts(gL, gG, S, False)[1])              # no dispersion asked for


@pytest.mark.parametrize('gL, gG', WIDTHS)
def test_the_twin_has_the_derivatives_of_the_kernel(gL, gG):
    A, D = dk._voigt_parts(gL, gG, S, True)
    h = STEP * (gL + gG)              # the line's width: the scale the kernel varies on in all three
    assert h < gG
    differences = [(_kernel(gL + h, gG, S) - _kernel(gL - h, gG, S)) / (2 * h),
                   (_kernel(gL, gG + h, S) - _kernel(gL, gG - h, S)) / (2 * h),
                   (_kernel(gL, gG, S + h) - _kernel(gL, gG, S - h)) / (2 * h)]
    for row, reference in zip((1, 2, 3), differences):
        scale = np.max(np.abs(reference))
        assert scale > 0
        assert np.max(np.abs(A[row] + 1j * D[row] - reference)) < ACCURACY * scale


def test_no_gaussian_width_the_derivative_in_it_is_the_one_sided_one():
    # At gG = 0 the kernel has only widths above: the one-sided difference of
    # second order, (-3 f(0) + 4 f(h) - f(2h))/(2h), for both of its parts.
    gL = 0.097
    h = STEP * gL
    A, D = dk._voigt_parts(gL, 0.0, S, True)
    one_sided = (-3 * _kernel(gL, 0.0, S) + 4 * _kernel(gL, h, S) - _kernel(gL, 2 * h, S)) / (2 * h)
    assert np.max(np.abs(one_sided.real)) > 0 and np.max(np.abs(one_sided.imag)) > 0
    assert np.max(np.abs(A[2] + 1j * D[2] - one_sided)) < ACCURACY * np.max(np.abs(_kernel(gL, 0.0, S))) / gL


def test_the_far_wings_of_a_narrow_line_are_finite():
    for part in dk._voigt_parts(1e-3, 5e-4, np.array([-1e3, -1.0, 0.0, 1.0, 1e3]), True):
        assert np.all(np.isfinite(part))


def _differences(f, p, free):
    rows = []
    for k in free:
        h = STEP * max(abs(p[k]), 1.0)
        up, down = list(p), list(p)
        up[k] += h
        down[k] -= h
        rows.append((np.asarray(f(up)) - np.asarray(f(down))) / (2 * h))
    return np.array(rows)


def _assert_jacobian(on_duals, on_numbers, p, free):
    """The model written with the twins, on Duals, has the value and the
    Jacobian (by differences) of the same model written with the kernels."""
    y = on_duals(dn.seed(p, free))
    plain = np.asarray(on_numbers(list(p)))
    scale = max(np.max(np.abs(plain)), 1.0)
    assert np.max(np.abs(dn.value(y) - plain)) < 100 * EPS * scale
    J = dn.jacobian(y, free)
    reference = _differences(on_numbers, p, free)
    assert np.max(np.abs(J - reference)) < ACCURACY * max(np.max(np.abs(reference)), scale)
    assert np.all(np.any(J != 0, axis=1))           # every fitted parameter moves the model
    return J


def _singlet(voigt):
    """TImod's 'Singlet' at one node: T, delta, the two widths; p[4] not fitted."""
    def transmission(p):
        return np.exp(-np.pi * m5.NAT_WIDTH / 2 * abs(p[0]) * voigt(abs(p[2]), abs(p[3]) * p[4], S - p[1]))
    return transmission


def test_a_scalar_component_written_with_the_twin():
    J = _assert_jacobian(_singlet(dk.Voigt), _singlet(m5.Voigt), [3.0, 0.4, 0.2, 0.15, 1.0], [0, 1, 2, 3])
    assert J.shape == (4, len(S))


def test_plain_arguments_give_a_plain_array():
    out = dk.Voigt_complex(0.2, 0.15, S, 1.0, 1j)
    assert isinstance(out, np.ndarray) and out.dtype == complex
    assert isinstance(dk.Voigt(0.2, 0.15, S), np.ndarray) and dk.Voigt(0.2, 0.15, S).dtype == float


def test_positions_on_a_grid():
    # SCDW's use: every energy against every grid node, summed with weights
    grid, weights = np.array([-2.0, -0.5, 1.5]), np.array([0.2, 0.5, 0.3])

    def line(voigt):
        return lambda p: (voigt(p[0], p[1], S[:, None] - (grid * p[2])[None, :], 1.0, 1j) * weights[None, :]).sum(axis=1)
    J = _assert_jacobian(line(dk.Voigt_complex), line(m5.Voigt_complex), [0.2, 0.15, 1.1], [0, 1, 2])
    assert J.dtype == complex


def _mm_perp(mx, my):
    """models._mm_perp for direction cosines that may be Duals."""
    return dn.array([[mx * mx, mx * my], [mx * my, my * my]], dtype=complex)


def _smat(voigt, mm_perp, p):
    """The cross-section matrix of a thick component as TImod builds one: two
    lines with the texture-blended matrices of a fitted axis."""
    mx, my, mz = m5._axis_xyz(p[4], p[5])
    mm = mm_perp(mx, my)
    Mh_A = m5._texture_blend(1.5 * (m5._I2 - mm), p[6])
    Mh_B = m5._texture_blend(0.5 * m5._I2 + 1.5 * mm, p[6])
    VoiA = voigt(abs(p[2]), abs(p[3]), S - (p[1] + 1.0), 1.0, 1j)
    VoiB = voigt(abs(p[2]), abs(p[3]), S - (p[1] - 1.0), 1.0, 1j)
    return np.pi * m5.NAT_WIDTH / 2 * (0.5 * abs(p[0])) * (VoiA[:, None, None] * Mh_A[None, :, :]
                                                         + VoiB[:, None, None] * Mh_B[None, :, :])


RHO = (0.95, 0.05)                    # an SMS beam polarised to 0.9


def _layer(voigt, mm_perp, expm_neg, readout):
    """The detected transmission of that component: the amplitude matrix
    expm(-Smat/2) and the polarised readout."""
    return lambda p: readout(expm_neg(_smat(voigt, mm_perp, p) / 2), *RHO)


def _models_readout(Tprod, rho11, rho22):
    return m5._polarised_readout(Tprod, 0, rho11 - rho22)


ON_NUMBERS = _layer(m5.Voigt_complex, m5._mm_perp, m5._expm_neg, _models_readout)
# The models' own matrix code on Duals, operation by operation, and the twins
AS_WRITTEN = _layer(dk.Voigt_complex, _mm_perp, m5._expm_neg, _models_readout)
TWINS = _layer(dk.Voigt_complex, _mm_perp, dk.expm_neg, dk.readout)

# T, delta, gL, gG, theta, phi, the texture A
LAYER = [8.0, 0.3, 0.2, 0.15, 55.0, 20.0, 0.6]


@pytest.mark.parametrize('on_duals', [AS_WRITTEN, TWINS], ids=['as written', 'twins'])
def test_the_matrix_path_on_duals(on_duals):
    J = _assert_jacobian(on_duals, ON_NUMBERS, LAYER, list(range(len(LAYER))))
    # not vacuous: the polarised readout is not the scalar transmission
    scalar = np.exp(-np.pi * m5.NAT_WIDTH / 2 * 0.5 * LAYER[0]
                    * (m5.Voigt(0.2, 0.15, S - 1.3) + m5.Voigt(0.2, 0.15, S + 0.7)))
    assert np.max(np.abs(ON_NUMBERS(LAYER) - scalar)) > 1e-2
    assert J.dtype == float


@pytest.mark.parametrize('on_duals', [AS_WRITTEN, TWINS], ids=['as written', 'twins'])
def test_a_powder_layer_has_its_derivative_in_the_texture(on_duals):
    # A = 0: Smat is a multiple of the identity at every energy, where
    # _expm_neg's square root is 0 and the closed form takes its limit.
    powder = LAYER[:6] + [0.0]
    z = _eigenvalue_split(_smat(m5.Voigt_complex, m5._mm_perp, powder) / 2)
    assert np.max(np.abs(z)) < EPS
    J = _assert_jacobian(on_duals, ON_NUMBERS, powder, [0, 1, 6])
    assert np.all(np.isfinite(J))


def _eigenvalue_split(M):
    """z = s^2 - det of 2x2 matrices: the square of half the difference of
    their eigenvalues, what the twin's series switches on."""
    s = 0.5 * (M[:, 0, 0] + M[:, 1, 1])
    return s * s - (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0])


def test_the_twin_joins_its_series_to_its_closed_form():
    # The texture taken through the switch: spectra whose energies all use the
    # series, and spectra with energies on both sides of it.
    in_series, across = 0, 0
    for A in (1e-7, 1e-4, 3e-3, 0.03, 0.1, 0.3):
        p = LAYER[:6] + [A]
        below = np.abs(_eigenvalue_split(_smat(m5.Voigt_complex, m5._mm_perp, p) / 2)) < dk._SERIES_BELOW
        in_series += np.all(below)
        across += np.any(below) and not np.all(below)
        _assert_jacobian(TWINS, ON_NUMBERS, p, [0, 1, 4, 6])
    assert in_series >= 2 and across >= 2


def test_the_expm_twin_on_matrices_that_are_not_hermitian():
    rng = np.random.default_rng(5)
    n, k = 40, 3
    Sig = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
    dSig = rng.normal(size=(n, 2, 2, k)) + 1j * rng.normal(size=(n, 2, 2, k))
    T, dT = dk._expm_neg_parts(Sig, dSig)
    reference = m5._expm_neg(Sig)
    assert np.max(np.abs(T - reference)) < 100 * EPS * np.max(np.abs(reference))
    for j in range(k):
        along = (m5._expm_neg(Sig + STEP * dSig[..., j]) - m5._expm_neg(Sig - STEP * dSig[..., j])) / (2 * STEP)
        assert np.max(np.abs(dT[..., j] - along)) < ACCURACY * np.max(np.abs(along))
    assert not np.allclose(np.matmul(Sig, np.conjugate(np.transpose(Sig, (0, 2, 1)))),
                           np.matmul(np.conjugate(np.transpose(Sig, (0, 2, 1))), Sig))     # not normal either
