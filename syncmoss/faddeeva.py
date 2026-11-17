"""The Faddeeva function w(z) = exp(-z^2) erfc(-iz), Im z >= 0, by Weideman's
expansion, and the exact Voigt line built on it -- the line shape of
models.Voigt_c / Voigt_r under COMPLEX_VOIGT_METHOD = 'faddeeva'.

    w(z) = 2 p(Z) / (L - iz)^2 + 1 / (sqrt(pi) (L - iz)),   Z = (L + iz)/(L - iz),

p a polynomial of degree FADDEEVA_TERMS - 1 (J. A. C. Weideman, SIAM J. Numer.
Anal. 31 (1994) 1497). Its coefficients are a Fourier transform, made once
(coefficients); a point then costs one complex division and FADDEEVA_TERMS - 1
complex multiply-adds -- plain arithmetic, which voigt_line runs over chunks of
points, so the compiler makes vector instructions of it.

The Voigt line of Lorentzian FWHM gL and Gaussian FWHM gG, unit area, is

    V + iD = w(z) / (sigma sqrt(2 pi)),   z = (S + i gL/2) / (sigma sqrt 2),
    sigma = gG / sqrt(8 ln 2),

V its absorptive part and D the dispersive one (Kramers-Kronig partner of V),
D > 0 for S > 0; without a Gaussian (gG <= 0) the Lorentzian
(1/pi) / (gL/2 - iS). Im z = gL/(2 sigma sqrt 2) > 0 for every line with a
width, the half-plane the expansion is made for.

Its dual twin, the line with its derivatives in its parameters, is
dual_kernels._faddeeva_parts: the same steps, which tests/test_faddeeva.py
holds to this module. The coefficients reach the compiled code as ARGUMENTS
(voigt_line's A and L): numba stores a compiled function with the values of
the globals it read, checking only its own file, so a global of this module
read by a kernel of another one would keep its old value after a change here.
"""
import math

import numpy as np
from numba import njit

#: Terms of the expansion. The error against w computed in 50-440 digit decimal
#: arithmetic (an independent reference, 297 points with Im z from 1e-10 to
#: 1000 and |Re z| to 1e5), relative to |w|: 3e-4 at 8 terms, 4e-7 at 16, 4e-10
#: at 24, 3e-13 at 32, 7e-15 at 36, 8.6e-16 at 40 -- and 8-15e-16 for every
#: count from there to 128: 40 is where it reaches the rounding of the double
#: arithmetic. (scipy.special.wofz is right to 9e-15 there.)
FADDEEVA_TERMS = 40

_S2L2 = math.sqrt(8.0 * math.log(2.0))           # FWHM / sigma of a Gaussian
_SQRT_2 = math.sqrt(2.0)
_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

#: Points of a chunk of voigt_line: the arrays of one chunk stay in the
#: first-level cache
CHUNK = 64


def coefficients(n):
    """The coefficients of the expansion with *n* terms, the highest power
    first, and its scale L: a discrete Fourier transform of
    (L^2 + t^2) exp(-t^2) at t = L tan(theta/2), no table."""
    L = np.sqrt(n / np.sqrt(2.0))
    k = np.arange(-2 * n + 1, 2 * n)
    t = L * np.tan(k * np.pi / (4 * n))
    f = np.concatenate(([0.0], np.exp(-t ** 2) * (L ** 2 + t ** 2)))
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (4 * n)
    return a[n:0:-1].copy(), float(L)


FADDEEVA_A, FADDEEVA_L = coefficients(FADDEEVA_TERMS)


def faddeeva(z):
    """w(z) for Im z >= 0, any shape: the expansion in numpy, by Horner's
    rule (voigt_line's arithmetic, one array operation at a time)."""
    z = np.asarray(z, dtype=complex)
    lz = FADDEEVA_L - 1j * z
    Z = (FADDEEVA_L + 1j * z) / lz
    p = np.full(Z.shape, FADDEEVA_A[0], dtype=complex)
    for coefficient in FADDEEVA_A[1:]:
        p = p * Z + coefficient
    return 2.0 * p / lz ** 2 + (1.0 / _SQRT_PI) / lz


@njit(cache=True)
def voigt_line(gL, gG, S, A, L, out):
    """The exact Voigt line of Lorentzian FWHM gL and Gaussian FWHM gG (unit
    area) at the points S (1-D): out[k, 0] = V(S[k]) and, when out has a
    second column, out[k, 1] = D(S[k]), its dispersion. A, L: the expansion's
    coefficients and scale (FADDEEVA_A, FADDEEVA_L). gG <= 0: the Lorentzian.

    The divisions are products with the reciprocal of |.|^2, and the steps of
    the polynomial run over a chunk of points as real arrays: a loop without
    calls, which the compiler turns into vector instructions."""
    n = S.shape[0]
    with_d = out.shape[1] > 1
    gamma = 0.5 * gL
    if gG <= 0.0:
        # (1/pi) / (gamma - iS)
        for k in range(n):
            den = math.pi * (gamma * gamma + S[k] * S[k])
            out[k, 0] = gamma / den
            if with_d:
                out[k, 1] = S[k] / den
        return
    na = A.shape[0]
    s2 = gG / _S2L2 * _SQRT_2                   # sigma sqrt 2
    norm = 1.0 / (gG / _S2L2 * _SQRT_2PI)       # 1 / (sigma sqrt(2 pi))
    zi = gamma / s2                             # Im z
    a = L + zi                                  # L - iz = a - i Re z
    b = L - zi                                  # L + iz = b + i Re z
    ir = np.empty(CHUNK)                        # 1/(L - iz)
    ii = np.empty(CHUNK)
    Zr = np.empty(CHUNK)                        # Z = (L + iz)/(L - iz)
    Zi = np.empty(CHUNK)
    pr = np.empty(CHUNK)                        # the polynomial in Z
    pi_ = np.empty(CHUNK)
    for lo in range(0, n, CHUNK):
        m = min(n, lo + CHUNK) - lo
        for k in range(m):
            zr = S[lo + k] / s2
            q = 1.0 / (a * a + zr * zr)
            ir[k] = a * q
            ii[k] = zr * q
            Zr[k] = b * ir[k] - zr * ii[k]
            Zi[k] = b * ii[k] + zr * ir[k]
            pr[k] = A[0]
            pi_[k] = 0.0
        for t in range(1, na):
            c = A[t]
            for k in range(m):
                r = pr[k] * Zr[k] - pi_[k] * Zi[k] + c
                pi_[k] = pr[k] * Zi[k] + pi_[k] * Zr[k]
                pr[k] = r
        # w = 2 p / (L - iz)^2 + (1/sqrt(pi)) / (L - iz)
        if with_d:
            for k in range(m):
                i2r = ir[k] * ir[k] - ii[k] * ii[k]
                i2i = 2.0 * ir[k] * ii[k]
                out[lo + k, 0] = (2.0 * (pr[k] * i2r - pi_[k] * i2i) + ir[k] / _SQRT_PI) * norm
                out[lo + k, 1] = (2.0 * (pr[k] * i2i + pi_[k] * i2r) + ii[k] / _SQRT_PI) * norm
        else:
            for k in range(m):
                i2r = ir[k] * ir[k] - ii[k] * ii[k]
                i2i = 2.0 * ir[k] * ii[k]
                out[lo + k, 0] = (2.0 * (pr[k] * i2r - pi_[k] * i2i) + ir[k] / _SQRT_PI) * norm


def voigt(gL, gG, S):
    """V, the absorptive part of the exact Voigt line, at S (any shape)."""
    S = np.asarray(S, dtype=np.float64)
    out = np.empty(S.shape, dtype=np.float64)
    voigt_line(float(gL), float(gG), np.ascontiguousarray(S.reshape(-1)), FADDEEVA_A, FADDEEVA_L,
               out.reshape(-1, 1))
    return out


def voigt_complex(gL, gG, S):
    """V + iD, the exact Voigt line with its dispersion, at S (any shape)."""
    S = np.asarray(S, dtype=np.float64)
    out = np.empty(S.shape, dtype=np.complex128)
    voigt_line(float(gL), float(gG), np.ascontiguousarray(S.reshape(-1)), FADDEEVA_A, FADDEEVA_L,
               out.reshape(-1).view(np.float64).reshape(-1, 2))
    return out
