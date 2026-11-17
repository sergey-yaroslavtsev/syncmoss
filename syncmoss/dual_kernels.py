"""The array work of models.py for Duals (dual_numbers), in compiled twins.

numba cannot carry a Dual through a kernel, and numpy written for Duals
repeats every operation on the energy axis for every fit parameter. A twin
here does that work in one compiled pass over plain arrays:

  * Voigt, Voigt_complex -- the kernels' names and signatures. Their twin
    returns the value and its derivatives in the kernel's OWN arguments, and
    dual_numbers.chain makes the Dual of them, whatever fit parameters the
    arguments depend on; voigt_value is the value alone, in one pass;
  * _faddeeva_parts -- the exact Voigt of faddeeva.voigt_line (models.Voigt_c
    and Voigt_r under COMPLEX_VOIGT_METHOD 'faddeeva' / 'wofz') with its
    derivatives, as _voigt_parts has them; voigt_parts and line_value choose
    between the two by models.exact_voigt();
  * expm_neg, readout -- models._expm_neg and the readout of
    models._polarised_readout: the value and its derivatives along the
    derivatives of the matrix handed in.

  * _transmission -- an absorber given as a list of lines (models_dual), at
    the energies of a node: the line shapes with their derivatives, the
    lines' sum, the layers, the readout, a thickness distribution.

The twins repeat the formulas of models.py (numba cannot differentiate them),
so tests/test_dual_kernels.py holds each twin to models.py: the value to
rounding, the derivatives against central differences of the original.

A new model usually needs nothing here: its lines are Voigt, Blume or
relaxation shapes (models_dual, "ADDING A MODEL"). A new line SHAPE is a
function like _shape_blume -- F[0] the shape, F[1 + i] its derivative in its
parameter i, written by hand -- with a code after SLOPE, its number of
parameters in _COUNT, its branch in _transmission, and its test against
central differences in tests/test_dual_kernels.py.
"""
import math

import numpy as np
from numba import njit

import syncmoss.faddeeva as faddeeva
from syncmoss.constants import LINE_RATIO_25, LINE_RATIO_34
from syncmoss.dual_numbers import Dual, chain, value


# =============================================================================
# The pseudo-Voigt of models.Voigt / models.Voigt_complex
# (Ida, Ando & Toraya, doi.org/10.1107/S0021889800010219)
# =============================================================================
# Four unit-area shapes -- Lorentzian L, Gaussian G, intermediate I and
# hyperbolic P -- of the widths W_k = w_k(ro) (gL + gG) and the weights
# c_k(ro), ro = gL/(gL + gG). The polynomials in ro of models.Voigt, lowest
# power first:
_W_L = np.array([-0.42179, -1.25693, 10.30003, -23.45651, 29.14158, -16.50453, 3.19974])
_W_G = np.array([0.66000, 0.15021, -1.24984, 4.74052, -9.48291, 8.48252, -2.95553])
_W_I = np.array([1.19913, 1.43021, -15.36331, 47.06071, -73.61822, 57.92559, -17.80614])
_W_P = np.array([1.10186, -0.47745, -0.68688, 2.76622, -4.55466, 4.05475, -1.26571])
_C_L = np.array([-0.30165, -1.38927, 9.31550, -24.10743, 34.96491, -21.18862, 3.70290])
_C_I = np.array([0.25437, -0.14107, 3.23653, -11.09215, 22.10544, -24.12407, 9.76947])
_C_P = np.array([1.01579, 1.50429, -9.21815, 23.59717, -39.71134, 32.83023, -10.02142])
# A shape is written in its scale N_k = K_k W_k (L, G, I, P):
_K_G = 1 / (2 * np.sqrt(2 * np.log(2)))
_K = np.array([0.5, _K_G, 1 / (2 * (2 ** (2 / 3) - 1) ** (1 / 2)), 1 / (2 * np.log(2 ** (1 / 2) + 1))])


@njit(cache=True)
def _poly(c, x):
    """sum_k c[k] x^k and its derivative in x (Horner, the derivative carried along)."""
    f = 0.0
    f_x = 0.0
    for k in range(len(c) - 1, -1, -1):
        f_x = f_x * x + f
        f = f * x + c[k]
    return f, f_x


@njit(cache=True)
def _voigt_parts(gL, gG, S, dispersion):
    """The two real parts of models.Voigt_complex(gL, gG, S, C1, C2) -- the
    absorptive A and the dispersive D, Voigt_complex = C1 A + C2 D -- each with
    its derivatives: rows (value, d/dgL, d/dgG, d/dS) over the points of S.
    D is left zero unless *dispersion*.

    With f_k the shapes and f' = df/dS,
        A = sum_k c_k f_k,
        D = (2/gL) sum_k c_k (S f_k + delt^2 f_k'),
    delt the Gaussian sigma of gG (what the kernel subtracts from S in each
    term is -f'/f times delt^2). A shape of unit area is f = phi(S/W)/W, so
    its derivatives in the width follow from those in S:
        df/dW = -(f + S f')/W,    df'/dW = -(2 f' + S f'')/W.
    """
    g = gL + gG
    ro = gL / g
    # Widths W = w g and weights c of L, G, I, P; "_r": their derivative in ro
    w = np.empty(4)
    w_r = np.empty(4)
    c = np.empty(4)
    c_r = np.empty(4)
    P, P_r = _poly(_W_L, ro)
    w[0] = 1 - (1 - ro) * P
    w_r[0] = P - (1 - ro) * P_r
    P, P_r = _poly(_W_G, ro)
    w[1] = 1 - ro * P
    w_r[1] = -P - ro * P_r
    w[2], w_r[2] = _poly(_W_I, ro)
    w[3], w_r[3] = _poly(_W_P, ro)
    P, P_r = _poly(_C_L, ro)
    c[0] = ro * (1 + (1 - ro) * P)
    c_r[0] = 1 + (1 - 2 * ro) * P + ro * (1 - ro) * P_r
    P, P_r = _poly(_C_I, ro)
    c[2] = ro * (1 - ro) * P
    c_r[2] = (1 - 2 * ro) * P + ro * (1 - ro) * P_r
    P, P_r = _poly(_C_P, ro)
    c[3] = ro * (1 - ro) * P
    c_r[3] = (1 - 2 * ro) * P + ro * (1 - ro) * P_r
    c[1] = 1 - c[0] - c[2] - c[3]
    c_r[1] = -c_r[0] - c_r[2] - c_r[3]
    # In gL and gG: d ro/d gL = gG/g^2, d ro/d gG = -gL/g^2
    W = w * g
    N = W * _K
    W_L = w + w_r * gG / g
    W_G = w - w_r * gL / g
    c_L = c_r * gG / g ** 2
    c_G = -c_r * gL / g ** 2
    delt = gG * _K_G
    d2 = delt * delt

    A = np.empty((4, len(S)))
    D = np.zeros((4, len(S)))
    f = np.empty(4)         # the shapes at one point,
    f1 = np.empty(4)        # their first
    f2 = np.empty(4)        # and second derivatives in S
    for i in range(len(S)):
        s = S[i]
        q = s * s + N[0] * N[0]
        f[0] = N[0] / np.pi / q
        f1[0] = -2 * s * f[0] / q
        f2[0] = 2 * f[0] / q * (4 * s * s / q - 1)
        f[1] = np.exp(-s * s / (2 * N[1] * N[1])) / N[1] / np.sqrt(2 * np.pi)
        f1[1] = -s * f[1] / (N[1] * N[1])
        f2[1] = f[1] / (N[1] * N[1]) * (s * s / (N[1] * N[1]) - 1)
        r = N[2] * N[2] + s * s
        f[2] = N[2] * N[2] / (2 * r * np.sqrt(r))
        f1[2] = -3 * s * f[2] / r
        f2[2] = 3 * f[2] / r * (5 * s * s / r - 1)
        # sech^2 = 4m/(1 + m)^2 and |tanh| = (1 - m)/(1 + m), m = exp(-2|S|/N):
        # nothing overflows in the far wings of a narrow line
        m = np.exp(-2 * np.abs(s) / N[3])
        th = (1 - m) / (1 + m)
        if s < 0:
            th = -th
        f[3] = 2 * m / (1 + m) ** 2 / N[3]
        f1[3] = -2 * f[3] * th / N[3]
        f2[3] = 2 * f[3] / (N[3] * N[3]) * (3 * th * th - 1)

        a = a_L = a_G = a_S = 0.0
        b = b_L = b_G = b_S = 0.0
        for k in range(4):
            f_W = -(f[k] + s * f1[k]) / W[k]
            a += c[k] * f[k]
            a_L += c_L[k] * f[k] + c[k] * f_W * W_L[k]
            a_G += c_G[k] * f[k] + c[k] * f_W * W_G[k]
            a_S += c[k] * f1[k]
            if dispersion:
                u = s * f[k] + d2 * f1[k]
                u_W = s * f_W - d2 * (2 * f1[k] + s * f2[k]) / W[k]
                b += c[k] * u
                b_L += c_L[k] * u + c[k] * u_W * W_L[k]
                b_G += c_G[k] * u + c[k] * (u_W * W_G[k] + 2 * delt * _K_G * f1[k])
                b_S += c[k] * (f[k] + s * f1[k] + d2 * f2[k])
        A[0, i] = a
        A[1, i] = a_L
        A[2, i] = a_G
        A[3, i] = a_S
        if dispersion:
            D[0, i] = 2 / gL * b
            D[1, i] = 2 / gL * b_L - D[0, i] / gL
            D[2, i] = 2 / gL * b_G
            D[3, i] = 2 / gL * b_S
    return A, D


@njit(cache=True)
def voigt_value(gL, gG, S):
    """models.Voigt(gL, gG, S) -- _voigt_parts' value alone -- in one pass
    over the points of S (1-D), its factors made once: half the time of
    models.Voigt's array expression, for a caller that computes a line many
    times (the line fits of spectrum_reference and components_search). The
    value is models.Voigt's to rounding."""
    g = gL + gG
    ro = gL / g
    P, _ = _poly(_W_L, ro)
    N0 = (1 - (1 - ro) * P) * g * _K[0]
    P, _ = _poly(_W_G, ro)
    N1 = (1 - ro * P) * g * _K[1]
    P, _ = _poly(_W_I, ro)
    N2 = P * g * _K[2]
    P, _ = _poly(_W_P, ro)
    N3 = P * g * _K[3]
    P, _ = _poly(_C_L, ro)
    cL = ro * (1 + (1 - ro) * P)
    P, _ = _poly(_C_I, ro)
    cI = ro * (1 - ro) * P
    P, _ = _poly(_C_P, ro)
    cP = ro * (1 - ro) * P
    cG = 1 - cL - cI - cP
    # the four shapes of _voigt_parts, L G I P, as factors times what varies
    aL, aG, aI, aP = cL * N0 / np.pi, cG / N1 / np.sqrt(2 * np.pi), cI * N2 * N2 / 2, cP * 2 / N3
    qG, qP = 1 / (2 * N1 * N1), 2 / N3
    out = np.empty(len(S))
    for i in range(len(S)):
        s2 = S[i] * S[i]
        r = N2 * N2 + s2
        m = np.exp(-qP * np.abs(S[i]))           # sech^2 as 4m/(1 + m)^2: no overflow far out
        out[i] = aL / (s2 + N0 * N0) + aG * np.exp(-s2 * qG) + aI / (r * np.sqrt(r)) + aP * m / ((1 + m) * (1 + m))
    return out


# =============================================================================
# The exact Voigt of faddeeva.voigt_line (COMPLEX_VOIGT_METHOD 'faddeeva')
# =============================================================================
# faddeeva.py's constants, written out here: a kernel keeps the values of the
# globals it read from another file (see faddeeva.py) -- the coefficients come
# in as arguments for that reason
_F_S2L2 = math.sqrt(8.0 * math.log(2.0))
_F_SQRT_2 = math.sqrt(2.0)
_F_SQRT_PI = math.sqrt(math.pi)
_F_SQRT_2PI = math.sqrt(2.0 * math.pi)
_F_CHUNK = 64
#: |z| from which the slopes of w are taken from its asymptotic series: below
#: it w' = -2 z w + 2i/sqrt(pi) and z w' + w lose |z|^2 of w's digits to
#: cancellation; above it the series' terms fall as |z|^-2 (12 terms: 2e-18 at
#: |z| = 15). Against w' in high-precision arithmetic: 1e-13 below, 5e-16
#: above (the time domain's derivatives, measured 2026-11-17).
Z_ASYMPTOTIC = 15.0
_ASYMPTOTIC_TERMS = 12


@njit(cache=True)
def _w_slopes(z, w):
    """(w'(z), z w'(z) + w(z)) for w = w(z)."""
    if abs(z) < Z_ASYMPTOTIC:
        wp = -2.0 * z * w + 2j / _F_SQRT_PI
        return wp, z * wp + w
    # w ~ (i/sqrt(pi)) sum_n c_n z^-(2n+1), c_n = (2n-1)!!/2^n
    u = 1.0 / (z * z)
    c = 1.0
    un = 1.0 + 0j
    s1 = 0j               # sum (2n+1) c_n u^n
    s2 = 0j               # sum 2n c_n u^n
    for n in range(_ASYMPTOTIC_TERMS):
        s1 += (2 * n + 1) * c * un
        s2 += 2 * n * c * un
        c *= (2 * n + 1) / 2.0
        un *= u
    pref = 1j / _F_SQRT_PI
    return -pref * s1 * u, -pref * s2 / z


@njit(cache=True)
def _faddeeva_parts(gL, gG, S, A, L, dispersion):
    """_voigt_parts of the exact Voigt of faddeeva.voigt_line, by its steps
    (the value is voigt_line's bit for bit): the absorptive part V and the
    dispersive D, each with its derivatives, rows (value, d/dgL, d/dgG, d/dS)
    over the points of S; D left zero unless *dispersion*. A, L: the
    expansion's coefficients and scale (faddeeva.FADDEEVA_A, FADDEEVA_L).

    F = V + iD = w(z) / (sigma sqrt(2 pi)) with z = (S + i gL/2) / (sigma sqrt 2)
    and sigma = gG / sqrt(8 ln 2) depends on S + i gL/2 alone, so
        dF/dS = w'(z) / (sigma sqrt 2 sigma sqrt(2 pi)),   dF/dgL = (i/2) dF/dS,
        dF/dsigma = -(w + z w') / (sigma^2 sqrt(2 pi)),     dF/dgG = dF/dsigma / sqrt(8 ln 2).
    Without a Gaussian (gG <= 0) the Lorentzian (1/pi)/(gL/2 - iS); its
    derivative in gG is 0, the Voigt being even in sigma."""
    n = len(S)
    Ar = np.zeros((4, n))
    Dr = np.zeros((4, n))
    gamma = 0.5 * gL
    if gG <= 0.0:
        for k in range(n):
            den = np.pi * (gamma * gamma + S[k] * S[k])     # the value as voigt_line takes it
            u = gamma - 1j * S[k]
            F_S = 1j / (np.pi * u * u)
            F_L = 0.5j * F_S
            Ar[0, k] = gamma / den
            Ar[1, k] = F_L.real
            Ar[3, k] = F_S.real
            if dispersion:
                Dr[0, k] = S[k] / den
                Dr[1, k] = F_L.imag
                Dr[3, k] = F_S.imag
        return Ar, Dr
    # w at every point, as faddeeva.voigt_line takes it
    na = A.shape[0]
    sigma = gG / _F_S2L2
    s2 = gG / _F_S2L2 * _F_SQRT_2
    norm = 1.0 / (gG / _F_S2L2 * _F_SQRT_2PI)
    zi = gamma / s2
    a = L + zi
    b = L - zi
    wr = np.empty(n)
    wi = np.empty(n)
    ir = np.empty(_F_CHUNK)
    ii = np.empty(_F_CHUNK)
    Zr = np.empty(_F_CHUNK)
    Zi = np.empty(_F_CHUNK)
    pr = np.empty(_F_CHUNK)
    pi_ = np.empty(_F_CHUNK)
    for lo in range(0, n, _F_CHUNK):
        m = min(n, lo + _F_CHUNK) - lo
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
        for k in range(m):
            i2r = ir[k] * ir[k] - ii[k] * ii[k]
            i2i = 2.0 * ir[k] * ii[k]
            wr[lo + k] = 2.0 * (pr[k] * i2r - pi_[k] * i2i) + ir[k] / _F_SQRT_PI
            wi[lo + k] = 2.0 * (pr[k] * i2i + pi_[k] * i2r) + ii[k] / _F_SQRT_PI
    # the line and its derivatives
    for k in range(n):
        w = wr[k] + 1j * wi[k]
        wp, zwpw = _w_slopes(S[k] / s2 + 1j * zi, w)
        F_S = wp * (norm / s2)
        F_L = 0.5j * F_S
        F_G = zwpw * (-norm / (sigma * _F_S2L2))
        Ar[0, k] = wr[k] * norm
        Ar[1, k] = F_L.real
        Ar[2, k] = F_G.real
        Ar[3, k] = F_S.real
        if dispersion:
            Dr[0, k] = wi[k] * norm
            Dr[1, k] = F_L.imag
            Dr[2, k] = F_G.imag
            Dr[3, k] = F_S.imag
    return Ar, Dr


def voigt_parts(gL, gG, S, dispersion, exact):
    """_voigt_parts (the pseudo-Voigt), or for *exact* (models.exact_voigt())
    _faddeeva_parts (the exact Voigt), with the expansion's coefficients."""
    if exact:
        return _faddeeva_parts(gL, gG, S, faddeeva.FADDEEVA_A, faddeeva.FADDEEVA_L, dispersion)
    return _voigt_parts(gL, gG, S, dispersion)


def line_value(gL, gG, S, exact):
    """voigt_value (the pseudo-Voigt) at the points S (1-D), or for *exact*
    the exact Voigt's (faddeeva.voigt)."""
    return faddeeva.voigt(gL, gG, S) if exact else voigt_value(gL, gG, S)


def Voigt_complex(gL, gG, S, C1, C2):
    """models.Voigt_complex for widths gL, gG (numbers) and positions S (an
    array of any shape) that may be Duals; C1 and C2 are plain numbers. A Dual
    when an argument is one, else the plain array."""
    s = np.asarray(value(S), dtype=float)
    A, D = _voigt_parts(float(value(gL)), float(value(gG)), np.ascontiguousarray(s).ravel(), C2 != 0)
    parts = (C1 * A + C2 * D if C2 != 0 else C1 * A).reshape((4,) + s.shape)
    return chain(parts[0], parts[1:], (gL, gG, S))


def Voigt(gL, gG, S):
    """models.Voigt for arguments that may be Duals."""
    return Voigt_complex(gL, gG, S, 1.0, 0.0)


# =============================================================================
# The matrix path: models._expm_neg and the readout of models._polarised_readout
# =============================================================================
# Both take Duals as they are written in models.py, operation by operation on
# arrays over the energies and the parameters; here each is one pass.

# Where the closed form of GAMMA below has lost eps/|z| of its digits and the
# first term left out of its series, of the order z^4, is far below eps
_SERIES_BELOW = np.finfo(float).eps ** (1 / 5)


@njit(cache=True)
def _expm_neg_parts(Sig, dSig):
    """T = expm(-Sig) for 2x2 matrices Sig (n, 2, 2), complex, and its
    derivatives dT (n, 2, 2, k) along the directions dSig (n, 2, 2, k).

    The value is models._expm_neg's closed form: with M = -Sig = s + N, N
    traceless and N^2 = z (z = s^2 - det M),
        T = e^s (ch + sc N),    ch = cosh sqrt(z),  sc = sinh sqrt(z)/sqrt(z).
    Along a direction X = dM the integral int_0^1 e^((1-t)M) X e^(tM) dt is
        dT = e^s { sc X + [sc tr(X)/2 + GAMMA tr(N X)] N
                        + [sc tr(N X)/2 + (ch - sc) tr(X)/2] },
        GAMMA = (ch - sc)/(2 z) = 1/6 + z/60 + z^2/1680 + z^3/90720 + ...
    (ch, sc and GAMMA are entire in z: nothing is singular where the two
    eigenvalues meet, as in a powder, whose Sig is a multiple of the identity).
    """
    n, k = dSig.shape[0], dSig.shape[3]
    T = np.empty((n, 2, 2), dtype=np.complex128)
    dT = np.empty((n, 2, 2, k), dtype=np.complex128)
    for e in range(n):
        a = -Sig[e, 0, 0]
        b = -Sig[e, 0, 1]
        c = -Sig[e, 1, 0]
        d = -Sig[e, 1, 1]
        s = 0.5 * (a + d)
        z = s * s - (a * d - b * c)
        disc = np.sqrt(z)
        if np.abs(disc) < 1e-12:                  # models._expm_neg's own limit
            sc = 1.0 + 0j
        else:
            sc = np.sinh(disc) / disc
        ch = np.cosh(disc)
        es = np.exp(s)
        T[e, 0, 0] = es * (ch + sc * (a - s))
        T[e, 0, 1] = es * (sc * b)
        T[e, 1, 0] = es * (sc * c)
        T[e, 1, 1] = es * (ch + sc * (d - s))
        if np.abs(z) < _SERIES_BELOW:
            gamma = 1 / 6 + z * (1 / 60 + z * (1 / 1680 + z / 90720))
        else:
            gamma = (ch - sc) / (2 * z)
        for j in range(k):
            x00 = -dSig[e, 0, 0, j]
            x01 = -dSig[e, 0, 1, j]
            x10 = -dSig[e, 1, 0, j]
            x11 = -dSig[e, 1, 1, j]
            tX = x00 + x11
            tNX = (a - s) * x00 + b * x10 + c * x01 + (d - s) * x11
            along_N = sc * tX / 2 + gamma * tNX
            along_1 = sc * tNX / 2 + (ch - sc) * tX / 2
            dT[e, 0, 0, j] = es * (sc * x00 + along_N * (a - s) + along_1)
            dT[e, 0, 1, j] = es * (sc * x01 + along_N * b)
            dT[e, 1, 0, j] = es * (sc * x10 + along_N * c)
            dT[e, 1, 1, j] = es * (sc * x11 + along_N * (d - s) + along_1)
    return T, dT


def expm_neg(Sig):
    """models._expm_neg for a stack of 2x2 matrices (n, 2, 2) that is a Dual."""
    T, dT = _expm_neg_parts(np.ascontiguousarray(Sig.v, dtype=complex), np.ascontiguousarray(Sig.d, dtype=complex))
    return Dual(T, dT, Sig.idx)


def readout(Tprod, rho11, rho22):
    """The detected fraction rho11 ||Tprod e1||^2 + rho22 ||Tprod e2||^2 of
    models._polarised_readout, for the amplitude operator Tprod (n, 2, 2) that
    is a Dual: d|t|^2 = 2 Re(conj(t) dt), element by element."""
    weight = np.conjugate(Tprod.v) * np.array([rho11, rho22])
    return Dual(np.real(np.sum(weight * Tprod.v, axis=(1, 2))),
                2 * np.real(np.einsum('nab,nabk->nk', weight, Tprod.d)), Tprod.idx)


# =============================================================================
# The line shapes of an absorber
# =============================================================================
# A line of an absorber is a SHAPE with its own parameters. A shape function
# returns F (1 + PARAMETERS, n), complex: F[0] the shape at the energies E and
# F[1 + i] its derivative in its i-th parameter. *sign* is
# models.DISPERSION_SIGN (0: no dispersion, the real shapes): a shape with the
# absorptive part V and the dispersive part D is V + 1j sign D.
VOIGT, BLUME, RELAX = 0, 1, 2            # RELAX + g: the group g = 0, 1, 2 of a Relax_MS
SLOPE = 5                                # a Voigt line's derivative in its position
PARAMETERS = 7                           # the most a shape has
_COUNT = np.array([3, 5, 6, 6, 6, 0])    # those a shape is differentiated in


@njit(cache=True)
def _shape_voigt(E, par, sign, dispersion, exact, fA, fL):
    """models.Voigt_r / Voigt_c at E - pos: par = (pos, gL, gG). *exact*: the
    exact Voigt (_faddeeva_parts, the expansion's coefficients fA and scale
    fL), else the pseudo-Voigt (_voigt_parts)."""
    if exact:
        A, D = _faddeeva_parts(par[1], par[2], E - par[0], fA, fL, dispersion)
    else:
        A, D = _voigt_parts(par[1], par[2], E - par[0], dispersion)
    F = np.zeros((1 + PARAMETERS, len(E)), dtype=np.complex128)
    for e in range(len(E)):
        F[0, e] = A[0, e] + 1j * sign * D[0, e]
        F[1, e] = -(A[3, e] + 1j * sign * D[3, e])
        F[2, e] = A[1, e] + 1j * sign * D[1, e]
        F[3, e] = A[2, e] + 1j * sign * D[2, e]
    return F


@njit(cache=True)
def _shape_slope(E, par, sign, dispersion, exact, fA, fL):
    """_shape_voigt's derivative in its position, as a shape: par = (pos, gL, gG).
    It is what the FIRST MOMENT of lines that stand at one position multiplies:
        sum_k I_k V(E - pos_k) = (sum_k I_k) V(E - pos) + (sum_k I_k (pos_k - pos)) dV/dpos + ...
    Such a moment is 0 where the lines coincide and only its derivatives count
    (hamiltonian_dual), so this shape has no derivatives of its own: a line of
    it must have the amplitude, or the matrix, 0."""
    F = _shape_voigt(E, par, sign, dispersion, exact, fA, fL)
    for e in range(len(E)):
        F[0, e] = F[1, e]
        F[1, e] = 0.0
        F[2, e] = 0.0
        F[3, e] = 0.0
    return F


@njit(cache=True)
def _shape_blume(E, par, sign):
    """models.Blume / Blume_c, the line of two states that exchange:
    par = (alf, s, L, W, R) -- half the difference alf and the mean s of the two
    states' positions (Blume's alf, and what it subtracts from V), the width L,
    the rate W and the ratio R of the populations. With a = E - s,
        a1 = 1j (a + alf) + L/2 + W,    a4 = 1j (a - alf) + L/2 + R W,
        f = (R a4 + a1 + 2 R W) / ((R + 1) (a1 a4 - R W^2)) / pi,
    and the shape is Re f - 1j sign Im f."""
    alf, s, L, W, R = par[0], par[1], par[2], par[3], par[4]
    F = np.zeros((1 + PARAMETERS, len(E)), dtype=np.complex128)
    for e in range(len(E)):
        a = E[e] - s
        a1 = 1j * (a + alf) + L / 2 + W
        a4 = 1j * (a - alf) + L / 2 + R * W
        det = a1 * a4 - R * W * W
        N = R * a4 + a1 + 2 * R * W
        den = (R + 1) * det * np.pi
        f = N / den
        # (dN - N ddet/det)/den for every parameter; R is in (R + 1) too
        f_alf = (1j * (1 - R) - N * 1j * (a4 - a1) / det) / den
        f_s = (-1j * (R + 1) + N * 1j * (a4 + a1) / det) / den
        f_L = ((R + 1) / 2 - N * (a4 + a1) / 2 / det) / den
        f_W = ((R + 1) ** 2 - N * (a4 + R * a1 - 2 * R * W) / det) / den
        f_R = (a4 + R * W + 2 * W - N * (a1 * W - W * W) / det) / den - f / (R + 1)
        F[0, e] = f.real - 1j * sign * f.imag
        F[1, e] = f_alf.real - 1j * sign * f_alf.imag
        F[2, e] = f_s.real - 1j * sign * f_s.imag
        F[3, e] = f_L.real - 1j * sign * f_L.imag
        F[4, e] = f_W.real - 1j * sign * f_W.imag
        F[5, e] = f_R.real - 1j * sign * f_R.imag
    return F


@njit(cache=True)
def _shape_relax(E, par, sign, group):
    """A group of lines of models._relax_MS_groups(_c), the many-state
    relaxation: par = (Sig, eps, Hv, W, R, alfa, S); group 0, 1, 2 its outer,
    middle and inner lines.

    The kernel eliminates, energy by energy, the tridiagonal system A y = 1 of
    the 2S + 1 states -- diagonal M33 + 1j M44, below it M11, above it M22 --
    and returns t = w.y, w the states' weights (as Re t - 1j sign Im t). Here
    the same elimination gives y and, with the transposed system, z = A^-T w;
    then for every parameter
        dt = dw.y - z.(dA y),
    and dA is a diagonal or tridiagonal matrix that costs nothing: all six
    derivatives for two eliminations."""
    Sig, eps, Hv, W, R, alfa = par[0], par[1], par[2], par[3], par[4], par[5]
    S = int(par[6])
    N = 2 * S + 1
    ratio = 1.0
    if group == 1:
        ratio = LINE_RATIO_25
    elif group == 2:
        ratio = LINE_RATIO_34
    side = 1.0 if group == 0 else -1.0          # -eps for the outer lines, +eps for the others

    # The rates between neighbouring states: M11 = R q exp(alfa gd), M22 its mirror
    q = np.empty(2 * S)
    gd = np.zeros(2 * S)
    for i in range(2 * S):
        u = i + 1 - S
        q[i] = -(S * (S + 1) - u * (u - 1))
        if i < S:
            gd[i] = np.cos(np.pi / 2 * (u / S) ** 2) ** 2 - np.cos(np.pi / 2 * ((u - 1) / S) ** 2) ** 2
    M11_R = q * np.exp(alfa * gd)               # dM11/dR
    M11 = R * M11_R
    M11_a = M11 * gd                            # dM11/dalfa
    M22, M22_R, M22_a = M11[::-1], M11_R[::-1], M11_a[::-1]
    M33 = np.empty(N)                           # W - what leaves the state
    M33_R = np.empty(N)
    M33_a = np.empty(N)
    for i in range(N):
        gone, gone_R, gone_a = 0.0, 0.0, 0.0
        if i > 0:
            gone += M11[i - 1]
            gone_R += M11_R[i - 1]
            gone_a += M11_a[i - 1]
        if i < N - 1:
            gone += M22[i]
            gone_R += M22_R[i]
            gone_a += M22_a[i]
        M33[i] = W - gone
        M33_R[i] = -gone_R
        M33_a[i] = -gone_a
    # The states' weights and their derivative in alfa
    level = np.empty(N)
    for i in range(N):
        level[i] = np.cos(np.pi / 2 * ((i - S) / S) ** 2) ** 2
    w = np.exp(-alfa * level)
    w = w / np.sum(w)
    w_a = w * (np.sum(w * level) - level) / 2 / np.pi
    w = w / 2 / np.pi

    F = np.zeros((1 + PARAMETERS, len(E)), dtype=np.complex128)
    pivot = np.empty(N, dtype=np.complex128)
    y = np.empty(N, dtype=np.complex128)
    z = np.empty(N, dtype=np.complex128)
    parts = np.empty(7, dtype=np.complex128)
    for e in range(len(E)):
        pivot[0] = M33[0] + 1j * (E[e] - ratio * Hv - Sig - side * eps)
        y[0] = 1.0
        z[0] = w[0]
        for i in range(1, N):
            pivot[i] = (M33[i] + 1j * (E[e] - ratio * Hv * (S - i) / S - Sig - side * eps)
                        - M11[i - 1] * M22[i - 1] / pivot[i - 1])
            y[i] = 1.0 - M11[i - 1] * y[i - 1] / pivot[i - 1]
            z[i] = w[i] - M22[i - 1] * z[i - 1] / pivot[i - 1]
        y[N - 1] = y[N - 1] / pivot[N - 1]
        z[N - 1] = z[N - 1] / pivot[N - 1]
        for i in range(N - 2, -1, -1):
            y[i] = (y[i] - M22[i] * y[i + 1]) / pivot[i]
            z[i] = (z[i] - M11[i] * z[i + 1]) / pivot[i]
        t = 0j
        zy = 0j                     # z.y
        zy_H = 0j                   # z.(the states' share of Hv) y
        t_R = 0j                    # z.(dA/dR y)
        t_a = 0j                    # z.(dA/dalfa y)
        w_y = 0j                    # dw/dalfa . y
        for i in range(N):
            t += w[i] * y[i]
            w_y += w_a[i] * y[i]
            zy += z[i] * y[i]
            zy_H += z[i] * y[i] * (S - i) / S
            t_R += z[i] * M33_R[i] * y[i]
            t_a += z[i] * M33_a[i] * y[i]
            if i > 0:
                t_R += z[i] * M11_R[i - 1] * y[i - 1]
                t_a += z[i] * M11_a[i - 1] * y[i - 1]
            if i < N - 1:
                t_R += z[i] * M22_R[i] * y[i + 1]
                t_a += z[i] * M22_a[i] * y[i + 1]
        parts[0] = t
        parts[1] = 1j * zy                      # Sig:  dM44 = -1
        parts[2] = 1j * side * zy               # eps:  dM44 = -side
        parts[3] = 1j * ratio * zy_H            # Hv:   dM44 = -ratio (S - i)/S
        parts[4] = -zy                          # W:    dM33 = 1
        parts[5] = -t_R
        parts[6] = w_y - t_a
        for i in range(7):
            F[i, e] = parts[i].real - 1j * sign * parts[i].imag
    return F


# =============================================================================
# An absorber of lines: everything TImod does at the energies of a node
# =============================================================================
@njit(cache=True)
def _assembled(S, dS, g_M, g_dM, take):
    """The cross-section matrix sum_g S_g M_g of the groups *take* and its
    derivatives: (n, 2, 2) and (n, 2, 2, k)."""
    n, k = S.shape[1], dS.shape[2]
    Smat = np.zeros((n, 2, 2), dtype=np.complex128)
    dSmat = np.zeros((n, 2, 2, k), dtype=np.complex128)
    for g in range(len(g_M)):
        if not take[g]:
            continue
        for e in range(n):
            for c in range(2):
                for b in range(2):
                    Smat[e, c, b] += S[g, e] * g_M[g, c, b]
                    for j in range(k):
                        dSmat[e, c, b, j] += dS[g, e, j] * g_M[g, c, b] + S[g, e] * g_dM[g, c, b, j]
    return Smat, dSmat


@njit(cache=True)
def _seen(T, dT, rho11, rho22):
    """The readout rho11 ||T e1||^2 + rho22 ||T e2||^2 and its derivatives."""
    n, k = dT.shape[0], dT.shape[3]
    seen = np.empty(n)
    dseen = np.empty((n, k))
    for e in range(n):
        seen[e] = (rho11 * (np.abs(T[e, 0, 0]) ** 2 + np.abs(T[e, 1, 0]) ** 2)
                   + rho22 * (np.abs(T[e, 0, 1]) ** 2 + np.abs(T[e, 1, 1]) ** 2))
        for j in range(k):
            dseen[e, j] = 2 * (rho11 * (np.conj(T[e, 0, 0]) * dT[e, 0, 0, j]
                                        + np.conj(T[e, 1, 0]) * dT[e, 1, 0, j]).real
                               + rho22 * (np.conj(T[e, 0, 1]) * dT[e, 0, 1, j]
                                          + np.conj(T[e, 1, 1]) * dT[e, 1, 1, j]).real)
    return seen, dseen


@njit(cache=True)
def _transmission(E, l_shape, l_par, l_amp, l_group, l_scaled, l_dpar, l_damp,
                  g_M, g_dM, g_layer, g_scaled, t_s, t_w, t_ds, t_dw, sign, exact, fA, fL, rho11, rho22):
    """The detected transmission r (n,) of an absorber at the energies E and
    dr (n, k), its derivatives in k parameters. The absorber is a list of lines,
    each a shape (l_shape, with its parameters l_par) times an amplitude l_amp:

      * a line of the group -1 is a thin component's: exp(-amplitude Re shape);
      * the lines of a group g >= 0 are summed, sum_l amplitude shape, and the
        sum times the group's 2x2 matrix g_M is added to the cross-section
        matrix Smat of the group's layer (g_layer, 0 first in the beam). A
        layer's amplitude operator is expm(-Smat/2), the layers' product is
        read out with the beam's rho11, rho22 (models._polarised_readout).

    With the patches of a thickness distribution (t_s their relative
    thicknesses, t_w their shares of the beam; one layer): the lines and groups
    marked *scaled* are multiplied by a patch's thickness, and the patches'
    transmissions are averaged (models._thickness_average).

    The "d" arguments are the derivatives in the k parameters, with one last
    axis; a line is differentiated only in the parameters in which one of its
    own values is not zero. A Voigt line is the exact Voigt where *exact*
    (_faddeeva_parts: fA, fL the expansion's coefficients and scale), else the
    pseudo-Voigt (models.exact_voigt).
    """
    n, k, G = len(E), l_damp.shape[1], len(g_M)
    act = np.empty(k, dtype=np.int64)       # the parameters the line in hand depends on
    tau = np.zeros((2, n))                  # the thin lines' depth: not scaled, scaled
    dtau = np.zeros((2, n, k))
    S = np.zeros((G, n), dtype=np.complex128)
    dS = np.zeros((G, n, k), dtype=np.complex128)
    for l in range(len(l_shape)):
        g = l_group[l]
        shape = l_shape[l]
        if shape == VOIGT:
            F = _shape_voigt(E, l_par[l], sign, g >= 0 and sign != 0, exact, fA, fL)
        elif shape == BLUME:
            F = _shape_blume(E, l_par[l], sign)
        elif shape == SLOPE:
            F = _shape_slope(E, l_par[l], sign, g >= 0 and sign != 0, exact, fA, fL)
        else:
            F = _shape_relax(E, l_par[l], sign, shape - RELAX)
        count = _COUNT[shape]
        na = 0
        for j in range(k):
            own = l_damp[l, j] != 0
            for i in range(count):
                own = own or l_dpar[l, i, j] != 0
            if own:
                act[na] = j
                na += 1
        a = l_amp[l]
        if g < 0:
            c = l_scaled[l]
            for e in range(n):
                tau[c, e] += a * F[0, e].real
                for m in range(na):
                    j = act[m]
                    slope = 0.0
                    for i in range(count):
                        slope += F[1 + i, e].real * l_dpar[l, i, j]
                    dtau[c, e, j] += l_damp[l, j] * F[0, e].real + a * slope
        else:
            for e in range(n):
                S[g, e] += a * F[0, e]
                for m in range(na):
                    j = act[m]
                    slope_c = 0j
                    for i in range(count):
                        slope_c += F[1 + i, e] * l_dpar[l, i, j]
                    dS[g, e, j] += l_damp[l, j] * F[0, e] + a * slope_c

    r = np.zeros(n)
    dr = np.zeros((n, k))
    if len(t_s) == 0:
        # ---- no thickness distribution: the layers one after the other ----
        for e in range(n):
            r[e] = np.exp(-tau[0, e] - tau[1, e])
            for j in range(k):
                dr[e, j] = -r[e] * (dtau[0, e, j] + dtau[1, e, j])
        if G == 0:
            return r, dr
        Tprod = np.empty((n, 2, 2), dtype=np.complex128)
        dTprod = np.empty((n, 2, 2, k), dtype=np.complex128)
        started = False
        for layer in range(g_layer.max() + 1):
            take = g_layer == layer
            if not np.any(take):
                continue
            Smat, dSmat = _assembled(S, dS, g_M, g_dM, take)
            T, dT = _expm_neg_parts(Smat / 2, dSmat / 2)
            if not started:
                Tprod, dTprod = T, dT
                started = True
                continue
            for e in range(n):              # this layer after those before it: T Tprod
                p00, p01, p10, p11 = Tprod[e, 0, 0], Tprod[e, 0, 1], Tprod[e, 1, 0], Tprod[e, 1, 1]
                t00, t01, t10, t11 = T[e, 0, 0], T[e, 0, 1], T[e, 1, 0], T[e, 1, 1]
                for j in range(k):
                    q00, q01, q10, q11 = dTprod[e, 0, 0, j], dTprod[e, 0, 1, j], dTprod[e, 1, 0, j], dTprod[e, 1, 1, j]
                    dTprod[e, 0, 0, j] = dT[e, 0, 0, j] * p00 + dT[e, 0, 1, j] * p10 + t00 * q00 + t01 * q10
                    dTprod[e, 0, 1, j] = dT[e, 0, 0, j] * p01 + dT[e, 0, 1, j] * p11 + t00 * q01 + t01 * q11
                    dTprod[e, 1, 0, j] = dT[e, 1, 0, j] * p00 + dT[e, 1, 1, j] * p10 + t10 * q00 + t11 * q10
                    dTprod[e, 1, 1, j] = dT[e, 1, 0, j] * p01 + dT[e, 1, 1, j] * p11 + t10 * q01 + t11 * q11
                Tprod[e, 0, 0] = t00 * p00 + t01 * p10
                Tprod[e, 0, 1] = t00 * p01 + t01 * p11
                Tprod[e, 1, 0] = t10 * p00 + t11 * p10
                Tprod[e, 1, 1] = t10 * p01 + t11 * p11
        seen, dseen = _seen(Tprod, dTprod, rho11, rho22)
        for e in range(n):
            for j in range(k):
                dr[e, j] = dr[e, j] * seen[e] + r[e] * dseen[e, j]
            r[e] *= seen[e]
        return r, dr

    # ---- a thickness distribution: the patches' transmissions averaged ----
    Sa, dSa = _assembled(S, dS, g_M, g_dM, g_scaled == 0)
    Sb, dSb = _assembled(S, dS, g_M, g_dM, g_scaled == 1)
    Smat = np.empty((n, 2, 2), dtype=np.complex128)
    dSmat = np.empty((n, 2, 2, k), dtype=np.complex128)
    seen = np.ones(n)
    dseen = np.zeros((n, k))
    for patch in range(len(t_s)):
        s, w = t_s[patch], t_w[patch]
        if G > 0:
            for e in range(n):
                for c in range(2):
                    for b in range(2):
                        Smat[e, c, b] = 0.5 * (Sa[e, c, b] + s * Sb[e, c, b])
                        for j in range(k):
                            dSmat[e, c, b, j] = 0.5 * (dSa[e, c, b, j] + s * dSb[e, c, b, j]
                                                       + t_ds[patch, j] * Sb[e, c, b])
            T, dT = _expm_neg_parts(Smat, dSmat)
            seen, dseen = _seen(T, dT, rho11, rho22)
        for e in range(n):
            thin = np.exp(-tau[0, e] - s * tau[1, e])
            r[e] += w * thin * seen[e]
            for j in range(k):
                dthin = -thin * (dtau[0, e, j] + s * dtau[1, e, j] + t_ds[patch, j] * tau[1, e])
                dr[e, j] += t_dw[patch, j] * thin * seen[e] + w * (dthin * seen[e] + thin * dseen[e, j])
    return r, dr
