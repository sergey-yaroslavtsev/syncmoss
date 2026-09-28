# -*- coding: utf-8 -*-
"""
Ab-initio energy distribution S(E) of a 57FeBO3 Synchrotron Mossbauer Source.

This is the *theoretical* instrumental function of the SMS: the spectral density
of the beam leaving the iron borate crystal on the electronically forbidden,
nuclear-allowed (NNN)_rh Bragg reflection near the Neel temperature in a small
external magnetic field, as a function of the position on the rocking curve.

Nothing here is fitted.  The chain is:

  (1) exact hyperfine Hamiltonians (combined magnetic + quadrupole, at 90 deg)
      -> eigenvalues and eigenvectors                                    [1, 2]
  (2) M1 scattering amplitude operator f(b', b; E)      [2x2 in polarization]
  (3) pure-nuclear structure factor  F_H = f(+B_s) - f(-B_s)   (ODD part)
      forward structure factor       F_0 = f(+B_s) + f(-B_s)   (EVEN part)
  (4) exact two-beam dynamical diffraction -> R(E, theta)
  (5) S(E) = Tr[R rho_in R^+], averaged over mosaicity / setting error /
      temperature and field jitter

Two facts from (3) drive everything downstream and are what make this shape
different from the empirical Gaussian sum SYNCmoss has used until now:

  * the sum rule  sum_j c_j = 0  over the residues of chi_H (the Delta m = 0
    lines cancel identically in the pure-nuclear reflection), so chi_H ~ E^-2
    and therefore  S(E) ~ E^-4  for ANY thickness, angle, temperature and after
    any incoherent averaging.  A Gaussian instrumental function has no tails at
    all and a plain Lorentzian has E^-2 ones; both are wrong.
  * chi_H has exactly FOUR poles at every temperature and field, so in the
    weak-reflection limit S(E) is a rational function with 4 poles and 2 zeros.
    That is the ``rational_if`` primitive used as the storage form.

Units: energy [neV], length [Angstrom], angle [rad] internally; the public
helpers that SYNCmoss calls work in mm/s (``source_shape``, ``reduce_to_rational``).

All 57Fe constants come from syncmoss.constants -- deliberately, so the sextet
this module predicts coincides with the one models.py computes (the selftest
checks that to 1e-9 mm/s).  Note that SYNCmoss's nuclear magneton is DERIVED
from the alpha-Fe standard and sits 0.29 % below CODATA; using it here keeps one
energy scale across the program.

References
----------
[1] G. V. Smirnov, A. I. Chumakov, V. Potapkin, R. Rueffer, S. L. Popov,
    Phys. Rev. A 84, 053851 (2011)
[2] V. Potapkin et al., J. Synchrotron Rad. 19, 559 (2012)
[3] S. Yaroslavtsev, A. I. Chumakov, J. Synchrotron Rad. 29, 1329 (2022)
[4] G. V. Smirnov, Hyperfine Interact. 125, 91 (2000)
[5] A. M. Afanas'ev, Yu. Kagan, JETP Lett. 2, 81 (1965)
"""

from math import factorial, sqrt

import numpy as np

from syncmoss.constants import (
    NAT_WIDTH, ggr, gex, GAMMA0_NEV, MU_N_NEV_PER_T, MMPS_TO_NEV,
    LAMBDA_A, K_VEC_A, SIGMA0_A2, R_E_A,
)

# numpy >= 2 renamed trapz -> trapezoid; SYNCmoss still supports both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ======================================================================
#  1.  57FeBO3 MATERIAL / SETUP CONSTANTS
#
#  These are crystal and beamline properties, NOT 57Fe nuclear constants
#  (those live in syncmoss.constants and are imported above).
# ======================================================================

FEBO3_A_HEX = 4.626        # hexagonal a, Angstrom
FEBO3_C_HEX = 14.493       # hexagonal c, Angstrom
FEBO3_MU_LIN = 149.6       # photoabsorption coefficient at 14.4 keV, 1/cm
FEBO3_Z_CELL = ((26, 2), (5, 2), (8, 6))   # (Z, count) per rhombohedral cell
FEBO3_T_NEEL = 75.20       # Neel temperature in the ESRF setup, deg C

# Platelet thickness. The real crystal is 200-300 um; 250 is used because it is
# the middle of that range AND BECAUSE IT DOES NOT MATTER. In Bragg geometry the
# beam penetrates about one extinction depth -- 0.4 um at B_s = 2.4 T, 4 um at
# 0.2 T -- so everything past ~30 um is dark and never contributes. Measured:
# 35 um and 2000 um give identical spectra to six significant figures, at the
# rocking peak AND at the dip, at every field in the series. Photoabsorption
# (1/mu = 67 um) never enters for the same reason. Do not fit this.
DEFAULT_THICKNESS_UM = 250.0
# Lamb-Moessbauer factor of FeBO3 at the operating temperature (~348 K).
#
# NOT a guess and NOT a free parameter: it follows from the Debye model and a
# MEASURED Debye temperature. Lyubutin et al. (2022) give theta_D = 440 +- 6 K
# for FeBO3, and
#
#   -ln f = (3 E_R)/(2 k_B theta_D) [ 1 + 4 (T/theta_D)^2 int_0^{theta_D/T}
#                                         x dx/(e^x - 1) ]
#
# with E_R = E0^2/(2 M c^2) = 1.9583 meV gives f = 0.774 +- 0.005 at 348 K.
# (0.70 stood here before, which is 10 % low.)
#
# It MATTERS, contrary to the kinematical intuition: f_LM scales chi_H, chi_H
# sets the Darwin width and the extinction depth, and the reflectivity saturates
# on the plateau -- so scaling it changes WHICH energies sit on the plateau, i.e.
# the shape, and normalising the source afterwards does not undo that. Measured:
# at theta = 25 urad, B_s = 2.37 T the normalised FWHM runs 7.24 -> 10.87 Gamma_0
# as f_LM goes 0.40 -> 0.99. Only at theta = 0, where the reflectivity is
# off-plateau at every energy, is the shape nearly independent of it.
DEFAULT_F_LM = 0.774
DEFAULT_ENRICHMENT = 0.95      # 57Fe fraction
DEFAULT_MOSAIC_URAD = 5.0      # crystal slope error (FWHM)
DEFAULT_SETTING_URAD = 3.0     # angular positioning accuracy (FWHM)
DEFAULT_N_GAUSS = 7            # quadrature nodes of the Gaussian energy smear


# ======================================================================
#  2.  ANGULAR MOMENTUM
# ======================================================================

def _fact(x):
    return factorial(int(round(x)))


def clebsch_gordan(j1, m1, j2, m2, J, M):
    """<j1 m1; j2 m2 | J M>, Racah formula, real (Condon-Shortley) convention."""
    if abs(m1) > j1 + 1e-9 or abs(m2) > j2 + 1e-9 or abs(M) > J + 1e-9:
        return 0.0
    if abs(m1 + m2 - M) > 1e-9:
        return 0.0
    if J < abs(j1 - j2) - 1e-9 or J > j1 + j2 + 1e-9:
        return 0.0
    pre = sqrt((2 * J + 1) * _fact(J + j1 - j2) * _fact(J - j1 + j2)
               * _fact(j1 + j2 - J) / _fact(j1 + j2 + J + 1))
    pre *= sqrt(_fact(J + M) * _fact(J - M) * _fact(j1 - m1)
                * _fact(j1 + m1) * _fact(j2 - m2) * _fact(j2 + m2))
    kmin = int(round(max(0.0, j2 - J - m1, j1 - J + m2)))
    kmax = int(round(min(j1 + j2 - J, j1 - m1, j2 + m2)))
    s = 0.0
    for k in range(kmin, kmax + 1):
        s += (-1) ** k / (_fact(k) * _fact(j1 + j2 - J - k) * _fact(j1 - m1 - k)
                          * _fact(j2 + m2 - k) * _fact(J - j2 + m1 + k)
                          * _fact(J - j1 - m2 + k))
    return pre * s


def spin_operators(I):
    """Ix, Iy, Iz for spin I, in the basis m = I, I-1, ..., -I."""
    dim = int(round(2 * I + 1))
    m = np.array([I - i for i in range(dim)], dtype=float)
    Ip = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):                    # I+ |m> = c |m+1>
        mm = m[i]
        Ip[i - 1, i] = np.sqrt(I * (I + 1) - mm * (mm + 1))
    Im = Ip.conj().T
    return 0.5 * (Ip + Im), (Ip - Im) / 2.0j, np.diag(m).astype(complex), m


def m1_cartesian_operators():
    r"""
    Cartesian components (t_x, t_y, t_z) of the M1 transition operator between
    I_g = 1/2 (columns m_g = +1/2, -1/2) and I_e = 3/2 (rows m_e = 3/2 .. -3/2)
    with unit reduced matrix element:

        <I_e m_e| t_{1M} |I_g m_g> = <I_g m_g; 1 M | I_e m_e>
        t_x = (t_{-1} - t_{+1})/sqrt2,  t_y = i(t_{-1} + t_{+1})/sqrt2,  t_z = t_0

    Returns an array of shape (3, 4, 2).
    """
    mg = np.array([0.5, -0.5])
    me = np.array([1.5, 0.5, -0.5, -1.5])
    t = {}
    for M in (-1, 0, 1):
        A = np.zeros((4, 2), dtype=complex)
        for ie, mev in enumerate(me):
            for ig, mgv in enumerate(mg):
                if abs(mgv + M - mev) < 1e-9:
                    A[ie, ig] = clebsch_gordan(0.5, mgv, 1, M, 1.5, mev)
        t[M] = A
    return np.array([(t[-1] - t[+1]) / np.sqrt(2.0),
                     1j * (t[-1] + t[+1]) / np.sqrt(2.0),
                     t[0]])


_T_LAB = m1_cartesian_operators()


# ======================================================================
#  3.  HYPERFINE SITE  (exact diagonalisation)
# ======================================================================

class HyperfineSite:
    r"""
    H_g = -g_g mu_N B_s (m . I)                                    (I = 1/2)
    H_e =  (eQV_zz/12)[3 Iz^2 - I(I+1) + eta (Ix^2 - Iy^2)]
           - g_e mu_N B_s (m . I)                                  (I = 3/2)

    with z || c || V_zz and m in the basal plane, so the magnetic and quadrupole
    axes are at 90 deg.  Identical to models._ham_mono_core, whose Q parameter is
    the same eQV_zz (checked to 2e-15 mm/s in tests/test_sms_theory.py).

    PARAMETRISATION. The material constants of FeBO3 are the quadrupole COUPLING
    CONSTANT eQV_zz and the EFG asymmetry eta = (V_xx - V_yy)/V_zz -- both fixed
    properties of the crystal, the same at every temperature and every angle.
    They are passed here as

        dEQ = eQV_zz / 2          [neV]   ("the axial quadrupole splitting")
        eta                               dimensionless, 0 <= eta <= 1

    so that dEQ alone is the observable splitting when eta = 0. With eta != 0 the
    zero-field splitting becomes dEQ*sqrt(1 + eta^2/3); dEQ is NOT that splitting
    and should not be quoted as it.

    eta = 0 is exact for an ideal crystal -- the Fe site symmetry in FeBO3 is -3,
    which forces an axial EFG -- so a fit that wants eta != 0 is saying the site
    is distorted, or that something else is missing. It is offered because that
    is worth being able to test, not because it is expected.

    No perturbation theory: near T_N the magnetic splitting, the quadrupole
    splitting and Gamma_0 are all comparable.

    B_s [T], dEQ and shift [neV].
    """

    __slots__ = ("Ee", "Eg", "t", "dE")

    def __init__(self, B_s, m_hat, dEQ, shift=0.0, eta=0.0):
        Ixg, Iyg, Izg, _ = spin_operators(0.5)
        Ixe, Iye, Ize, _ = spin_operators(1.5)
        mx, my, mz = m_hat

        Hg = -ggr * MU_N_NEV_PER_T * B_s * (mx * Ixg + my * Iyg + mz * Izg)
        HQ = (dEQ / 6.0) * (3.0 * (Ize @ Ize) - 1.5 * 2.5 * np.eye(4))
        if eta:
            HQ = HQ + (dEQ / 6.0) * eta * (Ixe @ Ixe - Iye @ Iye)
        He = HQ - gex * MU_N_NEV_PER_T * B_s * (mx * Ixe + my * Iye + mz * Ize)

        self.Eg, Ug = np.linalg.eigh(Hg)
        self.Ee, Ue = np.linalg.eigh(He)
        # transition operator in the hyperfine eigenbasis
        self.t = np.einsum('bi,kij,ja->kba', Ue.conj().T, _T_LAB, Ug)
        self.dE = self.Ee[:, None] - self.Eg[None, :] + shift        # (4, 2)


def amplitude_matrix(site, b_out, b_in, E, f_LM, enrichment):
    r"""
    Coherent elastic M1 amplitude of ONE nucleus, in Angstrom, as a 2x2 matrix
    in polarization space for every energy:

        f(b',b;E) = -(k sigma_0 f_LM /4pi)(Gamma_0/2)(3/4)
                    * sum_{a,b}  [W(b')_{ba}]^*  W(b)_{ba}
                      / (E - dE_{ba} + i Gamma_0/2)

        W(v) = sum_i v_i t_i   with  b = k_hat x eps
                               (M1 couples to the MAGNETIC field of the wave)

    Normalisation: with no hyperfine splitting sum_{ab} |W(b)_{ba}|^2 =
    (4/3)|b|^2, so f -> -(k sigma_0 f_LM/4pi)(Gamma_0/2)(b'^*.b)/(E-E_0+i G/2)
    and sigma(E_0) = sigma_0 f_LM (optical theorem).  Checked in selftest().

    b_out, b_in : (2, 3) complex, rows = (sigma, pi) channels.
    E           : (NE,) [neV]
    returns     : (NE, 2, 2) complex, index [E, out, in]
    """
    W_out = np.einsum('pi,iba->pba', b_out, site.t)          # (2, 4, 2)
    W_in = np.einsum('pi,iba->pba', b_in, site.t)            # (2, 4, 2)
    num = np.einsum('oba,nba->onba', W_out.conj(), W_in)     # (out, in, 4, 2)
    den = E[None, None, :] - site.dE[:, :, None] + 1j * GAMMA0_NEV / 2.0
    res = np.einsum('onba,baE->Eon', num, 1.0 / den)
    pref = -(K_VEC_A * SIGMA0_A2 * f_LM * enrichment / (4.0 * np.pi)) \
        * (GAMMA0_NEV / 2.0) * 0.75
    return pref * res


# ======================================================================
#  4.  GEOMETRY
# ======================================================================

class Geometry:
    r"""
    Crystal frame: z || c || Q.  The (NNN) planes are parallel to the platelet
    surface, so the reflection is symmetric.  Scattering plane = x-z, sigma || y.

        k  = ( cos tB, 0, -sin tB)      k' = ( cos tB, 0, +sin tB)
        pi = k x sigma                  pi' = k' x sigma
        b  = k x eps                    (magnetic field of the wave)

    phi_m = azimuth of the staggered hyperfine field in the basal plane.
    phi_m = 0  ->  m IN the scattering plane == the SMS operating condition (a
                   small B_ext normal to the scattering plane forces this
                   through the Dzyaloshinskii-Moriya canting, L _|_ M).
    """

    def __init__(self, theta_B, phi_m=0.0):
        self.theta_B = theta_B
        t = theta_B
        self.k0 = np.array([np.cos(t), 0.0, -np.sin(t)])
        self.kH = np.array([np.cos(t), 0.0, +np.sin(t)])
        self.sig = np.array([0.0, 1.0, 0.0])
        pi0 = np.cross(self.k0, self.sig)
        piH = np.cross(self.kH, self.sig)
        self.b0 = np.array([pi0, np.cross(self.k0, pi0)], dtype=complex)
        self.bH = np.array([piH, np.cross(self.kH, piH)], dtype=complex)
        self.m_hat = np.array([np.cos(phi_m), np.sin(phi_m), 0.0])


# ======================================================================
#  5.  CRYSTAL, STRUCTURE FACTORS, SUSCEPTIBILITIES
# ======================================================================

class Crystal:
    """Iron borate platelet: lattice, reflection order, thickness, absorption.

    ``N_order`` is the (NNN)_rh order with N odd: 1 -> (111), 3 -> (333).
    """

    def __init__(self, a_hex=FEBO3_A_HEX, c_hex=FEBO3_C_HEX, N_order=1,
                 thickness=DEFAULT_THICKNESS_UM, f_LM=DEFAULT_F_LM,
                 enrichment=DEFAULT_ENRICHMENT, dw_H=1.0,
                 mu_lin=FEBO3_MU_LIN, Z_cell=FEBO3_Z_CELL):
        self.a_hex = float(a_hex)
        self.c_hex = float(c_hex)
        self.N_order = int(N_order)
        self.thickness = float(thickness)          # um
        self.f_LM = float(f_LM)
        self.enrichment = float(enrichment)
        self.dw_H = float(dw_H)                    # extra Debye-Waller at H
        self.mu_lin = float(mu_lin)                # 1/cm
        self.Z_cell = Z_cell

        V_hex = np.sqrt(3.0) / 2.0 * self.a_hex ** 2 * self.c_hex
        self.v_c = V_hex / 3.0                                 # A^3, 2 Fe inside
        self.d_spacing = self.c_hex / (3.0 * self.N_order)     # d((NNN)_rh)
        self.theta_B = float(np.arcsin(LAMBDA_A / (2.0 * self.d_spacing)))
        Ztot = sum(Z * n for Z, n in self.Z_cell)
        # convention: chi = 4 pi /(k^2 v_c) * F,  f_electron = -r_e Z
        chi_re = -(4.0 * np.pi * R_E_A / (K_VEC_A ** 2 * self.v_c)) * Ztot
        chi_im = (self.mu_lin * 1.0e-8) * LAMBDA_A / (2.0 * np.pi)
        self.chi0_el = chi_re + 1j * chi_im


def chi_matrices(cry, geo, B_s, dEQ, shift, E, eta=0.0):
    r"""
    chi_0, chi_H, chi_Hbar, each of shape (NE, 2, 2).

    Two Fe per rhombohedral cell, at (000) and (1/2,1/2,1/2).  For H = (NNN)_rh
    with N odd the geometric phase factor is exp(3 i pi N) = -1, hence

        F_H = e^{-W} [ f(+B_s) - f(-B_s) ]   (electronic part cancels EXACTLY)
        F_0 =         f(+B_s) + f(-B_s)

    i.e. the reflection is a perfect filter for the part of the scattering
    amplitude that is ODD in the staggered hyperfine field.
    """
    sp = HyperfineSite(B_s, +geo.m_hat, dEQ, shift, eta)
    sm = HyperfineSite(B_s, -geo.m_hat, dEQ, shift, eta)
    kw = dict(f_LM=cry.f_LM, enrichment=cry.enrichment)

    fH = amplitude_matrix(sp, geo.bH, geo.b0, E, **kw) \
        - amplitude_matrix(sm, geo.bH, geo.b0, E, **kw)          # 0 -> H
    fHb = amplitude_matrix(sp, geo.b0, geo.bH, E, **kw) \
        - amplitude_matrix(sm, geo.b0, geo.bH, E, **kw)          # H -> 0
    f0 = amplitude_matrix(sp, geo.b0, geo.b0, E, **kw) \
        + amplitude_matrix(sm, geo.b0, geo.b0, E, **kw)          # forward

    pref = 4.0 * np.pi / (K_VEC_A ** 2 * cry.v_c)
    chiH = pref * cry.dw_H * fH
    chiHb = pref * cry.dw_H * fHb
    chi0 = pref * f0 + cry.chi0_el * np.eye(2)[None, :, :]
    return chi0, chiH, chiHb


# ======================================================================
#  6.  EXACT TWO-BEAM DYNAMICAL DIFFRACTION
# ======================================================================

def reflectivity_matrix(chi0, chiH, chiHb, alpha, d_um, theta_B):
    r"""
    Exact 2x2 amplitude reflectivity of a plane-parallel crystal, symmetric
    Bragg case, for a stack of NE cases at once.

    Envelope (Takagi) equations for D = (D_0, D_H):

        dD/dz = i M D
        M = (k/2) [[ chi_0 / g0        ,  chi_Hbar / g0        ],
                   [ chi_H / gH        , (chi_0 - alpha) / gH  ]]
        g0 = +sin theta_B ,  gH = -sin theta_B

    Boundary conditions  D_0(0) = incident,  D_H(d) = 0.

    Solved by eigen-decomposition with the growing and decaying modes rescaled
    by exp(+-i lambda d), so the result stays finite even when d is thousands of
    extinction lengths (the nuclear extinction depth here is ~0.1 um while the
    crystal is ~35 um).

    alpha = (|k_0 + H|^2 - k^2)/k^2 ~= -2 dtheta sin(2 theta_B); it may be a
    scalar or a per-case array of length NE (that is how the mosaic average is
    batched into ONE eigen-decomposition).

    chi* : (NE, 2, 2);  returns (NE, 2, 2)
    """
    chi0 = np.asarray(chi0).reshape(-1, 2, 2)
    chiH = np.asarray(chiH).reshape(-1, 2, 2)
    chiHb = np.asarray(chiHb).reshape(-1, 2, 2)
    n = chi0.shape[0]

    if np.max(np.abs(chiH)) == 0.0:
        return np.zeros((n, 2, 2), dtype=complex)

    alpha = np.asarray(alpha, dtype=float).reshape(-1, 1, 1)

    d = d_um * 1.0e4                          # um -> A
    g0 = np.sin(theta_B)
    gH = -np.sin(theta_B)
    I2 = np.eye(2)[None, :, :]

    M = np.empty((n, 4, 4), dtype=complex)
    M[:, :2, :2] = (K_VEC_A / 2.0) * chi0 / g0
    M[:, :2, 2:] = (K_VEC_A / 2.0) * chiHb / g0
    M[:, 2:, :2] = (K_VEC_A / 2.0) * chiH / gH
    M[:, 2:, 2:] = (K_VEC_A / 2.0) * (chi0 - alpha * I2) / gH

    lam, P = np.linalg.eig(M)                          # (n,4), (n,4,4)
    order = np.argsort(-lam.imag, axis=1)              # decaying (Im>0) first
    lam = np.take_along_axis(lam, order, axis=1)
    P = np.take_along_axis(P, order[:, None, :], axis=2)

    Pd, Pg = P[:, :, :2], P[:, :, 2:]
    P0d, PHd = Pd[:, :2, :], Pd[:, 2:, :]
    P0g, PHg = Pg[:, :2, :], Pg[:, 2:, :]

    ud = np.exp(1j * lam[:, :2] * d)                   # |ud| <= 1
    vg = np.exp(-1j * lam[:, 2:] * d)                  # |vg| <= 1

    A = np.empty((n, 4, 4), dtype=complex)
    A[:, :2, :2] = PHd * ud[:, None, :]
    A[:, :2, 2:] = PHg
    A[:, 2:, :2] = P0d
    A[:, 2:, 2:] = P0g * vg[:, None, :]

    rhs = np.zeros((n, 4, 2), dtype=complex)
    rhs[:, 2:, :] = I2

    X = np.linalg.solve(A, rhs)
    cd, cg = X[:, :2, :], X[:, 2:, :]
    return PHd @ cd + PHg @ (vg[:, :, None] * cg)


def alpha_of_theta(theta_rad, theta_B, chi0_el):
    """
    Angular deviation parameter with theta measured from the REFRACTION-CORRECTED
    Bragg position, as in Yaroslavtsev & Chumakov (2022).  theta = 0 puts the
    centre of the electronic dispersion surface at alpha = 2 Re chi_0.
    Positive theta = larger angle of incidence = the side the SMS is operated on.
    """
    return 2.0 * np.real(chi0_el) - 2.0 * np.asarray(theta_rad) * np.sin(2.0 * theta_B)


# ======================================================================
#  7.  TOP LEVEL:  S(E) AT A GIVEN ROCKING-CURVE POSITION
# ======================================================================

class SMSParams:
    """Operating point of the source.

    B_s   : staggered hyperfine field [T] -- the temperature knob
    dEQ   : quadrupole splitting [neV].  Its SIGN is fixed (not fitted) by
            requiring the computed rocking curve to show the observed
            lower-left / higher-right double peak; its MAGNITUDE should come
            from a measured paramagnetic-phase FeBO3 spectrum.
    shift : isomer + second-order-Doppler shift of FeBO3 [neV]
    phi_m : azimuth of B_hf [rad]; 0 = in the scattering plane (SMS condition)
    """

    def __init__(self, B_s=0.50, dEQ=-0.20 * MMPS_TO_NEV, shift=0.0, phi_m=0.0,
                 eta=0.0,
                 pol_in=(1.0, 0.0), mosaic_fwhm_urad=DEFAULT_MOSAIC_URAD,
                 setting_fwhm_urad=DEFAULT_SETTING_URAD, dBs_rel_fwhm=0.0,
                 n_angle=11, n_field=1, dT_mK=0.0):
        self.B_s = float(B_s)
        self.dEQ = float(dEQ)
        self.shift = float(shift)
        self.eta = float(eta)
        self.phi_m = float(phi_m)
        self.pol_in = tuple(pol_in)
        self.mosaic_fwhm_urad = float(mosaic_fwhm_urad)
        self.setting_fwhm_urad = float(setting_fwhm_urad)
        self.dBs_rel_fwhm = float(dBs_rel_fwhm)
        self.dT_mK = float(dT_mK)
        self.n_angle = int(n_angle)
        self.n_field = int(n_field)


def _gauss_nodes(fwhm, n):
    if n <= 1 or fwhm <= 0:
        return np.array([0.0]), np.array([1.0])
    s = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    x = np.linspace(-2.5 * s, 2.5 * s, n)
    w = np.exp(-0.5 * (x / s) ** 2)
    return x, w / w.sum()


def _field_nodes(par):
    """Offsets dB and weights for the incoherent average over B_s.

    Two shapes, because they are two different physical situations:

    ``dBs_rel_fwhm``  a GAUSSIAN in the field itself -- symmetric, the right
                      model for field inhomogeneity across the footprint.

    ``dT_mK``         a Gaussian in TEMPERATURE, pushed through the calibrated
                      B_s(T). This is the one the SMS actually has: the crystal
                      sits a few tenths of a kelvin above T_N on a stage with a
                      gradient, and B_s ~ (T - T_N)^-p there. A symmetric spread
                      in T is therefore a strongly SKEWED, long-tailed spread in
                      B_s -- the cold side of the footprint gains far more field
                      than the warm side loses. With p ~ 1.8 and T - T_N ~ 0.2 K,
                      5 mK is already a few per cent, and the skew is what a
                      symmetric Gaussian in B_s cannot reproduce at any width.

    Both are supported; whichever is non-zero is used, and if both are set the
    temperature spread wins, because it is the physical one.
    """
    if par.dT_mK > 0 and par.B_s > 0:
        # Local form of the same power law, so it needs neither the clamped
        # range of Bs_from_temperature nor its inverse:
        #     B(T0 + d) = B_s * (1 + d/(T0 - T_N))^(-p)
        # with T0 - T_N recovered from B_s itself. Exact, and it stays valid
        # outside the temperatures the calibration was measured over.
        c = FEBO3_BS_CALIBRATION
        dT0 = (c['T_ref'] - c['T_N']) * (par.B_s / c['A']) ** (-1.0 / c['p'])
        dT, w = _gauss_nodes(par.dT_mK * 1e-3, par.n_field)
        r = 1.0 + dT / dT0
        ok = r > 1e-6
        if ok.sum() > 1:
            B = par.B_s * r[ok] ** (-c['p'])
            return B - par.B_s, w[ok] / w[ok].sum()
    return _gauss_nodes(par.dBs_rel_fwhm * par.B_s, par.n_field)


def sms_spectrum(E, theta_urad, cry, par, smear=True, return_amplitude=False):
    """
    S(E) = Tr[R rho_in R^+] at rocking-curve position ``theta_urad``, optionally
    averaged over mosaicity, setting error and B_s jitter.  ``E`` in neV.

    ``return_amplitude=True`` gives the UNSMEARED complex R(E) of shape
    (NE, 2, 2) at the nominal angle and field instead.

    The mosaic/setting average is batched into a single eigen-decomposition of
    an (n_angle*NE, 4, 4) stack -- that is what makes the shape cheap enough to
    evaluate inside the transmission integral.
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    NE = E.size
    geo = Geometry(cry.theta_B, par.phi_m)
    eps = np.asarray(par.pol_in, dtype=complex)
    eps = eps / np.linalg.norm(eps)

    if smear and not return_amplitude:
        fw = float(np.hypot(par.mosaic_fwhm_urad, par.setting_fwhm_urad))
        dth, wth = _gauss_nodes(fw, par.n_angle)
        dB, wB = _field_nodes(par)
    else:
        dth, wth = np.array([0.0]), np.array([1.0])
        dB, wB = np.array([0.0]), np.array([1.0])

    na = dth.size
    alpha = np.repeat(alpha_of_theta((theta_urad + dth) * 1e-6,
                                     cry.theta_B, cry.chi0_el), NE)

    S = np.zeros(NE, dtype=float)
    R_keep = None
    for db, wb in zip(dB, wB):
        chi0, chiH, chiHb = chi_matrices(cry, geo, par.B_s + db,
                                         par.dEQ, par.shift, E,
                                         getattr(par, 'eta', 0.0))
        if na > 1:
            chi0 = np.tile(chi0, (na, 1, 1))
            chiH = np.tile(chiH, (na, 1, 1))
            chiHb = np.tile(chiHb, (na, 1, 1))
        R = reflectivity_matrix(chi0, chiH, chiHb, alpha,
                                cry.thickness, cry.theta_B)
        if return_amplitude and R_keep is None:
            R_keep = R[:NE]
        out = np.einsum('Epq,q->Ep', R, eps)
        w = np.sum(np.abs(out) ** 2, axis=1).reshape(na, NE)
        S += wb * np.sum(wth[:, None] * w, axis=0)
    return R_keep if return_amplitude else S


def rocking_curve(theta_grid_urad, E, cry, par, smear=True):
    """Energy-integrated reflected intensity vs angle of incidence."""
    return np.array([_trapz(sms_spectrum(E, t, cry, par, smear), E)
                     for t in theta_grid_urad])


# --- calibrated B_s(T) -------------------------------------------------------
#
# Above T_N the staggered order is not spontaneous: it is INDUCED by the applied
# uniform field through the Dzyaloshinskii-Moriya coupling, so
# B_s ~ h * chi_staggered(T) ~ tau^-gamma. A critical power law is therefore the
# right functional form, and the mean-field gamma = 1 is far too shallow for the
# measured data.
#
# The constants below come from the ESRF (T, theta) study behind Yaroslavtsev &
# Chumakov (2022), 83 spectra at 9 temperatures, analysed with this module. Two
# independent observables were used:
#
#   * the measured COUNT RATE, fitted here. The reflectivity goes as |chi_H|^2
#     and chi_H is linear in B_s, so the intensity varies as ~B_s^2 over the
#     whole range -- the most uniform handle on B_s there is. With these
#     constants and ONE global constant converting the computed integrated
#     reflectivity into counts, the predicted intensity matches the measured one
#     to 7 % rms over a factor 270 in intensity (per-temperature ratios
#     0.93-1.05). Nothing else is adjusted; that is a test of the dynamical
#     diffraction, not a fit of it.
#   * the line SHAPE of the raw spectra, fitted separately. It gives
#     A = 0.4795, T_N = 75.3162, p = 2.873 -- different PARAMETERS but a curve
#     agreeing with this one to <= 10 % in B_s across the range. Take 10 % as the
#     real uncertainty of B_s(T), not the formal errors below.
#
# This is a CALIBRATION of one crystal in one magnet, not a material constant,
# and `T_N` is the apparent Neel temperature OF THE FIT -- it absorbs any offset
# of the temperature sensor (the setup's nominal value is FEBO3_T_NEEL = 75.20).
FEBO3_BS_CALIBRATION = {
    'A': 0.45673,       # B_s at T_ref [T]        (+- 0.008)
    'T_ref': 75.95,     # pivot of the power law [C]
    'T_N': 75.49191,    # apparent Neel temperature of the fit [C]  (+- 0.006)
    'p': 1.80601,       # critical exponent of chi_staggered        (+- 0.004)
    'T_lo': 75.675,     # range actually measured [C]
    'T_hi': 76.230,
}


def Bs_from_temperature(T_C, cal=None):
    r"""
    Staggered hyperfine field from the crystal temperature [deg C].

        B_s(T) = A * [ (T - T_N) / (T_ref - T_N) ] ^ (-p)

    Temperature enters the physics ONLY through B_s, so this is the whole of the
    temperature dependence of the instrumental function.

    Use it for a STARTING VALUE, not as a substitute for fitting B_s: it is a
    calibration of one crystal in one magnet (see FEBO3_BS_CALIBRATION) and it
    diverges as T -> T_N. Outside the measured range the result is the value at
    the nearer end -- extrapolating a critical power law is not meaningful.
    """
    c = dict(FEBO3_BS_CALIBRATION)
    if cal:
        c.update(cal)
    T = np.clip(np.asarray(T_C, dtype=float), c['T_lo'], c['T_hi'])
    tau = (T - c['T_N']) / (c['T_ref'] - c['T_N'])
    return float(c['A'] * np.maximum(tau, 1e-6) ** (-c['p'])) \
        if np.ndim(T_C) == 0 else c['A'] * np.maximum(tau, 1e-6) ** (-c['p'])


def temperature_from_Bs(B_s, cal=None):
    """Inverse of :func:`Bs_from_temperature` (clamped to the measured range)."""
    c = dict(FEBO3_BS_CALIBRATION)
    if cal:
        c.update(cal)
    tau = (np.asarray(B_s, dtype=float) / c['A']) ** (-1.0 / c['p'])
    T = c['T_N'] + tau * (c['T_ref'] - c['T_N'])
    return float(np.clip(T, c['T_lo'], c['T_hi'])) if np.ndim(B_s) == 0 \
        else np.clip(T, c['T_lo'], c['T_hi'])


# ======================================================================
#  8.  DERIVED OBSERVABLES
# ======================================================================

def line_metrics(E, S):
    """(FWHM, centre of the half-maximum points, area) -- all from S(E) itself."""
    S = np.asarray(S, dtype=float)
    if S.max() <= 0:
        return np.nan, np.nan, 0.0
    half = 0.5 * S.max()
    idx = np.where(S >= half)[0]
    i0, i1 = idx[0], idx[-1]

    def cross(i, j):
        if i == j or S[j] == S[i]:
            return E[i]
        return E[i] + (half - S[i]) * (E[j] - E[i]) / (S[j] - S[i])

    lo = cross(i0, i0 - 1) if i0 > 0 else E[0]
    hi = cross(i1, i1 + 1) if i1 < E.size - 1 else E[-1]
    return hi - lo, 0.5 * (hi + lo), float(_trapz(S, E))


def wide_energy_grid(e_lin=31.6, e_max=1.0e4, n_lin=2001, n_log=600):
    """
    Strictly increasing linear-core + log-wing grid, for tail work.

    The E^-4 wings carry real weight out to thousands of Gamma_0, so a plain
    linspace either truncates the integral or wastes points.  A non-monotonic
    concatenation silently corrupts every trapezoid integral downstream, so the
    result is asserted to be increasing.
    """
    E = np.concatenate([-np.logspace(np.log10(e_max), np.log10(e_lin), n_log),
                        np.linspace(-e_lin, e_lin, n_lin)[1:-1],
                        np.logspace(np.log10(e_lin), np.log10(e_max), n_log)])
    assert np.all(np.diff(E) > 0), "grid must be strictly increasing"
    return E


def tail_exponent(E, S, lo=1500.0, hi=9000.0):
    """Log-log slope of S(E) in the far wing.  Must be -4 (see module docstring)."""
    out = []
    for sgn in (+1, -1):
        m = (sgn * E > lo) & (sgn * E < hi) & (S > 0)
        out.append(np.polyfit(np.log(sgn * E[m]), np.log(S[m]), 1)[0]
                   if m.sum() > 5 else np.nan)
    return tuple(out)


# ======================================================================
#  8b.  ANALYTIC STRUCTURE: POLES, SUM RULES
# ======================================================================

def pole_residues(cry, geo, B_s, dEQ, shift=0.0, which="H", eta=0.0):
    r"""
    Exact pole-residue decomposition of the nuclear susceptibility,

        chi(E) = sum_j  c_j / (E - z_j),     z_j = E_j - i Gamma_0/2

    which="H"  -> chi_H, the sigma->pi channel of the pure-nuclear reflection
    which="0"  -> chi_0^sigma + chi_0^pi, the combination entering the two-beam
                  denominator

    Residues of the two sublattices are merged (they share the pole set, because
    H(-B) and H(+B) are unitarily equivalent).

    Returns (c, z) with duplicate poles combined and null residues dropped.
    The sum rule  sum_j c_j = 0  holds for which="H" and NOT for which="0".
    """
    sp = HyperfineSite(B_s, +geo.m_hat, dEQ, shift, eta)
    sm = HyperfineSite(B_s, -geo.m_hat, dEQ, shift, eta)
    amp = -(K_VEC_A * SIGMA0_A2 * cry.f_LM * cry.enrichment / (4 * np.pi)) \
        * (GAMMA0_NEV / 2.0) * 0.75
    pref = 4.0 * np.pi / (K_VEC_A ** 2 * cry.v_c)

    if which == "H":
        bo, bi, picks, sgn = geo.bH, geo.b0, [(1, 0)], -1.0
    else:
        bo, bi, picks, sgn = geo.b0, geo.b0, [(0, 0), (1, 1)], +1.0

    acc = {}
    for site, s_ in ((sp, 1.0), (sm, sgn)):
        Wo = np.einsum('pi,iba->pba', bo, site.t)
        Wi = np.einsum('pi,iba->pba', bi, site.t)
        num = np.einsum('oba,nba->onba', Wo.conj(), Wi)
        for (o, i) in picks:
            for b in range(4):
                for a in range(2):
                    key = round(float(site.dE[b, a]), 9)
                    acc[key] = acc.get(key, 0j) + s_ * pref * amp * num[o, i, b, a]
    if not acc:
        return np.array([]), np.array([])
    mx = max(abs(v) for v in acc.values())
    items = [(k, v) for k, v in sorted(acc.items()) if abs(v) > 1e-9 * mx]
    z = np.array([k - 1j * GAMMA0_NEV / 2 for k, _ in items])
    c = np.array([v for _, v in items])
    return c, z


def dressed_poles(cry, geo, B_s, dEQ, alpha, shift=0.0, lossless=False):
    r"""
    Poles of the weak-reflection reflectivity  R ~ chi_H / (alpha - 2 chi_0),
    i.e. the roots of

        (alpha - 2 chi_0^el) D(E) - sum_j d_j prod_{k != j} (E - z_k) = 0 .

    These are the *dressed* (radiatively shifted) resonances the instrumental
    function actually shows.  Width sum rule:

        sum_k g_k = n Gamma_0 / 2   exactly, when alpha - 2 chi_0^el is real,

    with photoabsorption pushing the sum ABOVE that value, the excess growing as
    the Bragg peak is approached -- so sum_k g_k >= n Gamma_0/2 is a hard bound.
    """
    d, z = pole_residues(cry, geo, B_s, dEQ, shift, which="0")
    chi0 = cry.chi0_el.real + 0j if lossless else cry.chi0_el
    at = alpha - 2.0 * chi0
    poly = at * np.poly(z)
    for j in range(len(z)):
        poly[1:] = poly[1:] - d[j] * np.poly(np.delete(z, j))
    return np.roots(poly)


# ======================================================================
#  9.  THE RATIONAL PRIMITIVE  (storage / evaluation form)
# ======================================================================

def rational_if(E, v1, g1, v2, g2):
    r"""
    Two-pole rational instrumental function, unit area, exact E^-4 tails:

        S(E) = C / ([(E-v1)^2+g1^2][(E-v2)^2+g2^2])
        C    = g1 g2 [(v1-v2)^2 + (g1+g2)^2] / (pi (g1+g2))

    Closed-form moments (both exact):
        centroid = (v1 g2 + v2 g1)/(g1+g2)
        variance = g1 g2 [(v1-v2)^2 + (g1+g2)^2]/(g1+g2)^2
    The third moment diverges logarithmically -- never use skewness here.

    v1 = v2 and g1 = g2 = Gamma_0/2 is the exact squared Lorentzian, the
    sub-natural 0.6436 Gamma_0 floor of the SMS line.
    """
    g1, g2 = abs(g1), abs(g2)
    C = g1 * g2 * ((v1 - v2) ** 2 + (g1 + g2) ** 2) / (np.pi * (g1 + g2))
    return C / (((E - v1) ** 2 + g1 ** 2) * ((E - v2) ** 2 + g2 ** 2))


def rational_moments(v1, g1, v2, g2):
    """(centroid, variance) of ``rational_if`` in closed form."""
    g1, g2 = abs(g1), abs(g2)
    return ((v1 * g2 + v2 * g1) / (g1 + g2),
            g1 * g2 * ((v1 - v2) ** 2 + (g1 + g2) ** 2) / (g1 + g2) ** 2)


def rational_tail_integral(E, v1, g1, v2, g2):
    r"""
    Closed-form upper-tail integral  int_E^inf rational_if(t) dt  (the total area
    is 1, so the lower tail is 1 - this).  Used to place the transmission-integral
    grid (models.limits) without a numerical quadrature.

    Partial fractions of the real integrand over the two conjugate pole pairs,

        1/([(x-v1)^2+g1^2][(x-v2)^2+g2^2])
            = (a1 (x-v1) + b1)/((x-v1)^2+g1^2) + (a2 (x-v2) + b2)/((x-v2)^2+g2^2)

    with, writing d = v1-v2 and D = (d^2+(g1+g2)^2)(d^2+(g1-g2)^2),

        a1 = -2d/D,  b1 = (d^2+g2^2-g1^2)/D,  a2 = +2d/D,  b2 = (d^2+g1^2-g2^2)/D.

    The antiderivative is (a/2) ln((x-v)^2+g^2) + (b/g) arctan((x-v)/g); the log
    terms cancel at infinity because a1 + a2 = 0 (that IS the E^-4 tail).

    D vanishes only for v1 == v2 AND g1 == g2, the exact squared Lorentzian,
    which is handled by its own closed form (the general expression then loses
    ~eps/delta digits to cancellation, hence the relative threshold).
    """
    g1, g2 = abs(g1), abs(g2)
    E = np.asarray(E, dtype=float)
    d = v1 - v2
    s = g1 + g2

    if d ** 2 + (g1 - g2) ** 2 <= 1.0e-12 * s ** 2:
        # squared Lorentzian: S = (2 g^3/pi)/((x-v)^2+g^2)^2
        g = 0.5 * s
        x = E - 0.5 * (v1 + v2)
        return 0.5 - (np.arctan(x / g) + g * x / (x ** 2 + g ** 2)) / np.pi

    C = g1 * g2 * (d ** 2 + s ** 2) / (np.pi * s)
    den = (d ** 2 + s ** 2) * (d ** 2 + (g1 - g2) ** 2)
    a1 = -2.0 * d / den
    b1 = (d ** 2 + g2 ** 2 - g1 ** 2) / den
    a2 = -a1
    b2 = (d ** 2 + g1 ** 2 - g2 ** 2) / den

    x1, x2 = E - v1, E - v2

    def _anti(x, a, b, g):
        return 0.5 * a * np.log(x ** 2 + g ** 2) + (b / g) * np.arctan(x / g)

    at_inf = (b1 / g1 + b2 / g2) * (np.pi / 2.0)
    at_E = _anti(x1, a1, b1, g1) + _anti(x2, a2, b2, g2)
    return C * (at_inf - at_E)


# ---------------------------------------------------------------- reduction

def rational_sum(E, params):
    r"""
    Sum of two-pole rational terms -- the storage form of the theoretical
    instrumental function::

        S(E) = sum_m w_m^2 * rational_if(E; v1_m, g1_m, v2_m, g2_m)

    ``params`` is the flat (v1, g1, v2, g2, w) x n_terms array.  Each term has
    unit area, so with sum_m w_m^2 = 1 the total area is 1 as well; that mirrors
    the amplitude convention of the legacy Gaussian sum, where the amplitudes
    are also normalised to unit sum of squares.

    A sum of such terms is the structurally correct object: the measured S(E) is
    an INCOHERENT average of |R|^2 over mosaicity / setting error / temperature,
    and every member of that average is a rational function with E^-4 tails.
    """
    p = np.asarray(params, dtype=float)
    E = np.asarray(E, dtype=float)
    out = np.zeros(np.shape(E), dtype=float)
    for m in range(len(p) // 5):
        v1, g1, v2, g2, w = p[5 * m:5 * m + 5]
        out = out + w ** 2 * rational_if(E, v1, g1, v2, g2)
    return out


def rational_sum_tail(E, params):
    """Upper-tail integral of :func:`rational_sum` (closed form, unit area)."""
    p = np.asarray(params, dtype=float)
    out = np.zeros(np.shape(np.asarray(E, dtype=float)), dtype=float)
    for m in range(len(p) // 5):
        v1, g1, v2, g2, w = p[5 * m:5 * m + 5]
        out = out + w ** 2 * rational_tail_integral(E, v1, g1, v2, g2)
    return out


def normalize_rational(params):
    """Rescale the weights of a rational-sum parameter vector to sum_m w_m^2 = 1.

    Returns ``(params, scale)`` where ``scale`` is the sum of squares BEFORE
    normalisation -- the caller multiplies the baseline by it, exactly as the
    legacy Gaussian path does.
    """
    p = np.array(params, dtype=float)
    n = len(p) // 5
    sc = float(np.sum(p[4::5][:n] ** 2))
    if sc <= 0:
        return p, 1.0
    p[4::5] = np.sqrt(p[4::5] ** 2 / sc)
    return p, sc


def rational_guesses(E, S):
    """Candidate single-term starting points for a numerically given S(E).

    The natural first guess -- a squared Lorentzian of the measured centre and
    FWHM -- sits exactly on the degenerate locus v1 = v2, g1 = g2.  Breaking that
    symmetry is a flat direction there, and Levenberg-Marquardt does not always
    leave it, so asymmetric starts (split in position, split in width, and both)
    are offered alongside it and the best result is kept.
    """
    w, c, _area = line_metrics(E, S)
    if not np.isfinite(w) or w <= 0:
        w = GAMMA0_NEV
    if not np.isfinite(c):
        c = 0.0
    g = 0.5 * w / 1.28719          # FWHM of the squared Lorentzian = 1.28719 g
    h = 0.5 * w
    return [np.array(q, dtype=float) for q in (
        (c, g, c, g, 1.0),                              # symmetric (L^2)
        (c - 0.4 * h, g, c + 0.4 * h, g, 1.0),          # split in position
        (c, 0.6 * g, c, 1.8 * g, 1.0),                  # split in width
        (c - 0.3 * h, 0.7 * g, c + 0.3 * h, 1.6 * g, 1.0),
    )]


def _rational_bounds(n_terms, centre, fwhm):
    """Box bounds for a rational-sum fit.

    Positions are kept within 30 FWHM of the measured centre and widths between
    0.1 and 30 natural widths (the sum rule puts the physical dressed half-widths
    at or above Gamma_0/2, so neither bound can cut off a real solution).  Loose
    grid-derived bounds let a term run off to the far wing, where it fits nothing
    and drags the whole reduction into a bad local minimum.
    """
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = GAMMA0_NEV
    if not np.isfinite(centre):
        centre = 0.0
    lo = np.array([-np.inf] * (5 * n_terms))
    hi = np.array([np.inf] * (5 * n_terms))
    for m in range(n_terms):
        for k in (0, 2):
            lo[5 * m + k] = centre - 30.0 * fwhm
            hi[5 * m + k] = centre + 30.0 * fwhm
        for k in (1, 3):
            lo[5 * m + k] = 0.1 * GAMMA0_NEV
            hi[5 * m + k] = max(30.0 * GAMMA0_NEV, 10.0 * fwhm)
    return np.array([lo, hi])


def reduce_to_rational(E, S, n_terms=2, p0=None, passes=3):
    """
    Reduce a numerically computed S(E) to the rational-sum storage form.

    This is the bridge between the two ways of describing the SMS: the dynamical
    calculation gives S(E) from (theta, B_s, dEQ) with no free parameters, and
    this collapses it to 5*n_terms numbers the transmission integral evaluates in
    closed form -- with the right E^-4 tails, unlike the Gaussian sum it replaces.

    Terms are added ONE AT A TIME (each new term is seeded by splitting the
    dominant one and broadening it) and every intermediate result is kept if it
    is the best so far: fitting 3 terms from scratch is badly conditioned and
    runs into the width bounds, fitting them incrementally is not.

    Returns ``(params, rms, max_dev)``, deviations relative to the peak of S.
    Uses SYNCmoss's own Levenberg-Marquardt (minimi_lib) rather than
    scipy.optimize, which is excluded from the distributed bundle.
    """
    import syncmoss.minimi_lib as mi

    E = np.asarray(E, dtype=float)
    S = np.asarray(S, dtype=float)
    area = float(_trapz(S, E))
    if area <= 0:
        raise ValueError("reduce_to_rational: S(E) has non-positive area")
    Sn = S / area
    peak = Sn.max()
    # Fit the shape scaled to unit peak: minimi_hi's Poisson-like 1/(|y|+1)
    # weighting then degenerates to an unweighted, peak-relative least squares.
    y = Sn / peak
    fwhm, centre, _ = line_metrics(E, Sn)

    def shape(x, pp):
        return rational_sum(x, pp) / peak

    def rms_of(pp):
        if pp is None or not np.all(np.isfinite(pp)):
            return np.inf
        return float(np.sqrt(np.mean(((rational_sum(E, pp) - Sn) / peak) ** 2)))

    def refine(p_start, npass):
        """Levenberg-Marquardt from one start; never returns a worse point."""
        p = np.array(p_start, dtype=float)
        bounds = _rational_bounds(len(p) // 5, centre, fwhm)
        p = np.clip(p, bounds[0], bounds[1])
        best, best_rms = p, rms_of(p)
        for _ in range(max(int(npass), 1)):
            try:
                p = mi.minimi_hi(shape, E, y, p, bounds=bounds,
                                 tau0=1e-4, MI=30, MI2=30, eps=1e-12)[0]
            except Exception:
                break
            r = rms_of(p)
            if not np.isfinite(r):
                break
            if r < best_rms:
                best_rms, best = r, np.array(p, dtype=float)
        return best, best_rms

    if p0 is not None:
        p, rms = refine(p0, passes)
    else:
        p, rms = None, np.inf
        for start in rational_guesses(E, Sn):
            p_try, rms_try = refine(start, passes)
            if rms_try < rms:
                p, rms = p_try, rms_try
        for _ in range(int(n_terms) - 1):
            # split the dominant term: half its weight moves to a broader copy
            k = int(np.argmax(np.abs(p[4::5])))
            v1, g1, v2, g2, w = p[5 * k:5 * k + 5]
            p_new = np.array(p, dtype=float)
            p_new[5 * k + 4] = w / np.sqrt(2.0)
            p_new = np.concatenate((p_new, [v1, 3.0 * g1, v2, 3.0 * g2,
                                            w / np.sqrt(2.0)]))
            p_try, rms_try = refine(p_new, passes)
            if rms_try >= rms:
                break            # the extra term did not help: keep what we have
            p, rms = p_try, rms_try

    p, _sc = normalize_rational(p)
    dev = (rational_sum(E, p) - Sn) / peak
    return p, float(np.sqrt(np.mean(dev ** 2))), float(np.max(np.abs(dev)))


# ======================================================================
# 10.  ABSORBER-SIDE DIAGNOSTICS (parameter-free tests of a candidate S)
# ======================================================================

def absorption_area(t_eff):
    r"""
    Total resonant absorption area for a single-line Lorentzian absorber and a
    DELTA source:

        A_delta(t) = (pi Gamma_0 / 2) t e^{-t/2} [I_0(t/2) + I_1(t/2)]

    By Fubini the measured area is independent of the source shape, so this
    determines the absorber thickness WITHOUT knowing the instrumental function
    -- and any candidate S(E) whose recovered area drifts with t is
    mis-normalised.
    """
    from scipy.special import i0e, i1e
    x = np.asarray(t_eff, dtype=float) / 2.0
    return (np.pi * GAMMA0_NEV / 2.0) * 2.0 * x * (i0e(x) + i1e(x))


def tmin_vs_thickness(E, S, t_list):
    r"""
    Transmission at the centre of a single absorber line versus its effective
    thickness -- the sharpest purely energy-domain test of the source tails.

    A thick line is a notch filter of half-width ~ sqrt(t); for a source tail
    S ~ |E|^-p the transmitted centre intensity scales as

        T_min(t) ~ t^{-(p-1)/2}   =>   slope -1.5 for p = 4
                                       slope -0.5 for a Lorentzian source
                                       exponential collapse for a Gaussian

    Returns (t_list, T_min, local log-log slope).
    """
    S = S / _trapz(S, E)
    T = []
    for t in t_list:
        a = t * (GAMMA0_NEV / 2) ** 2 / (E ** 2 + (GAMMA0_NEV / 2) ** 2)
        T.append(float(_trapz(S * np.exp(-a), E)))
    T = np.array(T)
    t_list = np.asarray(t_list, dtype=float)
    return t_list, T, np.gradient(np.log(T), np.log(t_list))


# ======================================================================
# 11.  SYNCmoss-FACING API  (mm/s in, mm/s out)
# ======================================================================

def crystal_and_params(theta_urad, B_s, dEQ_mmps, shift_mmps=0.0,
                       thickness_um=DEFAULT_THICKNESS_UM,
                       mosaic_urad=DEFAULT_MOSAIC_URAD,
                       setting_urad=DEFAULT_SETTING_URAD,
                       phi_m_deg=0.0, f_LM=DEFAULT_F_LM,
                       dBs_rel_fwhm=0.0, n_angle=11, n_field=1, N_order=1,
                       eta=0.0, dT_mK=0.0):
    """Build the (Crystal, SMSParams) pair from the physical parameters the
    NEW instrumental-function search fits, with velocities in mm/s."""
    cry = Crystal(N_order=int(N_order), thickness=float(thickness_um),
                  f_LM=float(f_LM))
    par = SMSParams(B_s=float(B_s), dEQ=float(dEQ_mmps) * MMPS_TO_NEV,
                    shift=float(shift_mmps) * MMPS_TO_NEV, eta=float(eta),
                    phi_m=np.deg2rad(float(phi_m_deg)),
                    mosaic_fwhm_urad=float(mosaic_urad),
                    setting_fwhm_urad=float(setting_urad),
                    dBs_rel_fwhm=float(dBs_rel_fwhm), dT_mK=float(dT_mK),
                    n_angle=int(n_angle), n_field=int(n_field))
    return cry, par


def source_shape(v_mmps, theta_urad, B_s, dEQ_mmps, shift_mmps=0.0, **kw):
    """S(v) of the source, v in mm/s.  Unnormalised (the transmission integral
    divides by its own numerical norm); use ``source_shape_normalised`` when an
    absolute density is wanted."""
    cry, par = crystal_and_params(theta_urad, B_s, dEQ_mmps, shift_mmps, **kw)
    return sms_spectrum(np.asarray(v_mmps, dtype=float) * MMPS_TO_NEV,
                        theta_urad, cry, par)


def source_shape_normalised(v_mmps, theta_urad, B_s, dEQ_mmps, shift_mmps=0.0,
                            grid=None, **kw):
    """S(v) in mm/s normalised to unit area, with the area taken on a wide
    linear-core + log-wing grid so the E^-4 tails are actually counted."""
    cry, par = crystal_and_params(theta_urad, B_s, dEQ_mmps, shift_mmps, **kw)
    if grid is None:
        grid = wide_energy_grid()
    Sg = sms_spectrum(grid, theta_urad, cry, par)
    area = float(_trapz(Sg, grid)) / MMPS_TO_NEV     # area in mm/s units
    S = sms_spectrum(np.asarray(v_mmps, dtype=float) * MMPS_TO_NEV,
                     theta_urad, cry, par)
    return S / area if area > 0 else S


def theory_to_rational(theta_urad, B_s, dEQ_mmps, shift_mmps=0.0, n_terms=2,
                       grid_mmps=None, **kw):
    """
    Compute the exact S(E) for one operating point and reduce it to the
    rational-sum storage form, in mm/s.

    Returns ``(params_mmps, rms, max_dev)``; ``params_mmps`` is the flat
    (v1, g1, v2, g2, w) x n_terms vector with all velocities in mm/s.
    """
    cry, par = crystal_and_params(theta_urad, B_s, dEQ_mmps, shift_mmps, **kw)
    if grid_mmps is None:
        # dense core (the line is a few Gamma_0 wide) plus wings out to 40 mm/s,
        # which is where the E^-4 tail has fallen ~6 orders below the peak.
        core = np.linspace(-1.2, 1.2, 1801)
        wing = np.concatenate([-np.logspace(np.log10(40.0), np.log10(1.2), 200),
                               np.logspace(np.log10(1.2), np.log10(40.0), 200)])
        grid_mmps = np.unique(np.concatenate([core, wing]))
    E = np.asarray(grid_mmps, dtype=float) * MMPS_TO_NEV
    S = sms_spectrum(E, theta_urad, cry, par)
    p, rms, mx = reduce_to_rational(E, S, n_terms=n_terms)
    p = np.array(p, dtype=float)
    p[0::5] /= MMPS_TO_NEV
    p[1::5] /= MMPS_TO_NEV
    p[2::5] /= MMPS_TO_NEV
    p[3::5] /= MMPS_TO_NEV
    return p, rms, mx


# ======================================================================
# 12.  THE INS ENCODING  (what INSexp.txt / #@INSexp actually holds)
# ======================================================================
#
# SYNCmoss stores the SMS instrumental function as a flat list of floats, and
# models.TImod consumes it with Met = 0.  Historically that list was always
# (width, position, amplitude) x n -- a sum of Gaussians.  The theoretical
# shapes are stored in the SAME slot and told apart by a sentinel first element
# that no legacy width parameter can take:
#
#   legacy   : [w, pos, amp] * n                              -> KIND_GAUSS
#   rational : [TAG, 1, (v1, g1, v2, g2, w) * n]              -> KIND_RATIONAL
#   physical : [TAG, 2, theta, B_s, dEQ, shift, d, mosaic,    -> KIND_PHYSICAL
#               setting, phi_m, f_LM, dBs_rel, na, nf, N]
#
# so every existing reader (INSexp.txt, the #@INSexp .dat header, the
# per-spectrum resolution, the metadata comparison) keeps working untouched.
# Velocities are in mm/s throughout, as everywhere else in SYNCmoss.

INS_TAG = -1.0e6
KIND_GAUSS = 0
KIND_RATIONAL = 1
KIND_PHYSICAL = 2

# order of the KIND_PHYSICAL payload, after [TAG, 2]
# NOTE ON ORDER: new fields go at the END, and decode_physical fills anything a
# shorter stored array does not carry from PHYS_DEFAULTS. An INSacc.txt written
# before gauss_fwhm existed therefore still decodes, as the unbroadened shape it
# described.
PHYS_FIELDS = ('theta_urad', 'B_s', 'dEQ', 'shift', 'thickness_um',
               'mosaic_urad', 'setting_urad', 'phi_m_deg', 'f_LM',
               'dBs_rel_fwhm', 'n_angle', 'n_field', 'N_order', 'eta',
               'gauss_fwhm', 'n_gauss', 'dT_mK')

PHYS_DEFAULTS = {
    'theta_urad': 70.0, 'B_s': 0.50, 'dEQ': -0.3228, 'shift': 0.0,
    'thickness_um': DEFAULT_THICKNESS_UM, 'mosaic_urad': DEFAULT_MOSAIC_URAD,
    'setting_urad': DEFAULT_SETTING_URAD, 'phi_m_deg': 0.0,
    'f_LM': DEFAULT_F_LM, 'dBs_rel_fwhm': 0.0, 'n_angle': 11.0,
    'n_field': 1.0, 'N_order': 1.0, 'eta': 0.0,
    'gauss_fwhm': 0.0, 'n_gauss': float(DEFAULT_N_GAUSS), 'dT_mK': 0.0,
}


def ins_kind(INS):
    """Which of the three instrumental-function forms ``INS`` holds."""
    a = np.atleast_1d(np.asarray(INS, dtype=float))
    if a.size >= 2 and a[0] == INS_TAG:
        k = int(round(a[1]))
        if k in (KIND_RATIONAL, KIND_PHYSICAL):
            return k
    return KIND_GAUSS


def encode_physical(**kw):
    """Flat KIND_PHYSICAL INS array from the physical parameters (mm/s, urad, T)."""
    vals = dict(PHYS_DEFAULTS)
    unknown = set(kw) - set(PHYS_FIELDS)
    if unknown:
        raise TypeError(f"encode_physical: unknown field(s) {sorted(unknown)}")
    vals.update(kw)
    return np.array([INS_TAG, float(KIND_PHYSICAL)]
                    + [float(vals[f]) for f in PHYS_FIELDS], dtype=float)


def decode_physical(INS):
    """dict of the physical parameters held by a KIND_PHYSICAL INS array."""
    a = np.atleast_1d(np.asarray(INS, dtype=float))
    if ins_kind(a) != KIND_PHYSICAL:
        raise ValueError("decode_physical: not a KIND_PHYSICAL instrumental function")
    body = a[2:]
    out = dict(PHYS_DEFAULTS)
    for i, f in enumerate(PHYS_FIELDS):
        if i < body.size:
            out[f] = float(body[i])
    return out


def encode_rational(params):
    """Flat KIND_RATIONAL INS array from a (v1, g1, v2, g2, w) x n vector [mm/s]."""
    p = np.asarray(params, dtype=float).ravel()
    if p.size == 0 or p.size % 5:
        raise ValueError("encode_rational: need 5 numbers per term")
    return np.concatenate(([INS_TAG, float(KIND_RATIONAL)], p))


def decode_rational(INS):
    """The (v1, g1, v2, g2, w) x n vector held by a KIND_RATIONAL INS array."""
    a = np.atleast_1d(np.asarray(INS, dtype=float))
    if ins_kind(a) != KIND_RATIONAL:
        raise ValueError("decode_rational: not a KIND_RATIONAL instrumental function")
    return a[2:]


# --- the simulated shape, tabulated once per parameter set --------------------
#
# models.TImod needs S at ONE velocity per integration node, and the transmission
# integral uses ~100 nodes; recomputing the dynamical diffraction per node would
# dominate the fit.  The shape is therefore evaluated once on a fixed grid (dense
# core + log wings), normalised to unit area, and interpolated afterwards.  The
# grid is FIXED, so the interpolated value at a fixed velocity is still a smooth
# function of the physical parameters -- which is what the Levenberg-Marquardt
# numerical derivatives need.  Outside the grid the exact E^-4 asymptote is used,
# so the tails stay right at any distance.

_SHAPE_CACHE = {}
_SHAPE_CACHE_MAX = 64

SHAPE_SCAN_PTS = 321        # coarse scan that locates the line
SHAPE_SCAN_MMPS = 4.0       # half-range of that scan [mm/s]
SHAPE_CORE_PTS = 1201       # dense points across the located line
SHAPE_CORE_FWHM = 12.0      # core half-width, in FWHM of the located line
SHAPE_WING_PTS = 150        # log-spaced points per side beyond the core
SHAPE_EDGE_MMPS = 60.0      # last tabulated velocity, from the line centre


def squared_lorentzian_limit():
    """Why the IDEAL source line is a squared Lorentzian of 0.6436 Gamma_0.

    Returns ``(fwhm_over_gamma0, tail_exponent)`` = (0.6436..., -4).

    The pure nuclear reflection exists because the two antiferromagnetic
    sublattices have opposite hyperfine fields and the (NNN)_rh reflection is
    forbidden for the electrons, so the nuclear structure factor is a
    DIFFERENCE of two scattering amplitudes,

        F_H(E) = f(E; +B_s) - f(E; -B_s).

    Each resonance contributes a Lorentzian AMPLITUDE c_j/(E - E_j + i*Gamma/2),
    and the difference flips the sign of every line's position, so the surviving
    lines come in pairs with equal and opposite residues. That is the filter
    theorem, and its arithmetic consequence is the sum rule sum_j c_j = 0
    (checked in ``selftest`` to 1e-15).

    Take the cleanest case -- one such pair, split by 2*delta:

        chi_H  =  c/(x - delta + i)  -  c/(x + delta + i)
               =  2 c delta / [ (x + i)^2 - delta^2 ]

    with x = 2(E - E_0)/Gamma. As the splitting closes (delta -> 0, i.e. B_s and
    dEQ both small compared with Gamma) this does NOT go to zero over a
    Lorentzian: it goes to a DERIVATIVE of one,

        chi_H  ->  2 c delta / (x + i)^2 ,

    and the reflected intensity, which off the Darwin plateau is proportional to
    |chi_H|^2, becomes

        S(E)  ~  1 / (1 + x^2)^2 ,

    a squared Lorentzian. Its FWHM is 2*sqrt(sqrt(2) - 1) in x, i.e.
    sqrt(sqrt(2) - 1) = 0.6436 of the natural width -- the source is SUB-natural
    not by any clever filtering of the wings but because a difference of two
    nearly equal amplitudes is a derivative, and a derivative is narrower than
    what it differentiates. The same square is what makes the tails go as E^-4
    instead of a Lorentzian's E^-2.

    Three things spoil it in the real source, all of them already parameters
    here: a finite splitting (B_s, dEQ) resolves the pair and widens/structures
    the line; dynamical diffraction on a thick crystal saturates the peak; and
    the incoherent averages (mosaic, field spread, ``gauss_fwhm``) smear it.
    None of them can make it NARROWER than 0.6436 Gamma_0, which is why a fit
    that wants a narrower source is telling you the absorber model is wrong.
    """
    return float(np.sqrt(np.sqrt(2.0) - 1.0)), -4


# How many of the FINEST grid steps the Gaussian sigma must span before the
# smear is done by interpolating one computed shape instead of re-evaluating it
# at every quadrature node. Four is comfortable: the interpolation error falls
# as (step/sigma)^2 and is already ~1e-4 of peak here, far below fit noise.
_SMEAR_INTERP_STEPS = 4.0

# Where ``shift`` sits in a KIND_PHYSICAL array: [INS_TAG, kind] + PHYS_FIELDS.
_SHIFT_INDEX = 2 + PHYS_FIELDS.index('shift')


def _eval_physical_sharp(v, ph, dshift=0.0):
    return np.maximum(np.asarray(source_shape(
        v, ph['theta_urad'], ph['B_s'], ph['dEQ'], ph['shift'] + dshift,
        thickness_um=ph['thickness_um'], mosaic_urad=ph['mosaic_urad'],
        setting_urad=ph['setting_urad'], phi_m_deg=ph['phi_m_deg'],
        f_LM=ph['f_LM'], dBs_rel_fwhm=ph['dBs_rel_fwhm'],
        dT_mK=ph.get('dT_mK', 0.0),
        n_angle=int(ph['n_angle']), n_field=int(ph['n_field']),
        N_order=int(ph['N_order']), eta=ph['eta']), dtype=float), 0.0)


def _eval_physical(v, ph):
    """S(v) including the Gaussian energy imperfection ``gauss_fwhm``.

    Everything incoherent that is NOT diffraction -- drive-velocity jitter and
    nonlinearity, the finite counting-time drift of the velocity zero, residual
    vibration, and any broadening of the resonance that is not a hyperfine
    effect -- adds up, by the central limit theorem, to one Gaussian in energy.
    The dynamical calculation has nothing that can produce it, so it has to be
    an explicit parameter; without it the ideal source is a SQUARED Lorentzian
    of 0.644 natural widths (see ``squared_lorentzian_limit``), which is
    narrower than anything ever measured.

    It is applied as a quadrature over energy offsets rather than an FFT,
    because S is tabulated on a grid that is deliberately not uniform (dense
    core, log wings). The E^-4 wings survive the smear -- convolving v^-4 with a
    compact kernel leaves v^-4.

    COST. Translating the spectrum is exactly a translation of ``shift``, so a
    node could be evaluated by re-running the whole dynamical calculation at
    ``shift + dk``. That is what this did, and it made every node cost a full
    evaluation: measured at n_angle = 11, one sharp shape is 112 ms and the
    7-node smear took 742 ms, dead linear in ``n_gauss``. Since the Gaussian
    imperfection stopped being pinned at zero that penalty applied to every
    Levenberg-Marquardt step of the instrumental-function search, which is why
    it became so slow.

    The nodes are shifts of ONE curve, so the shape is computed once and the
    nodes are interpolated from it -- the same trick sms_study/forward.py has
    used all along. Interpolation needs the grid to resolve the kernel, so it is
    used only when the Gaussian sigma spans several of the finest grid steps;
    below that the exact per-node evaluation is kept, where it is also cheap
    because the smear is then negligible anyway.
    """
    g = float(ph.get('gauss_fwhm', 0.0))
    n = int(ph.get('n_gauss', DEFAULT_N_GAUSS))
    if g <= 0 or n <= 1:
        return _eval_physical_sharp(v, ph)
    d, w = _gauss_nodes(g, n)
    vv = np.asarray(v, dtype=float)
    if vv.ndim == 1 and vv.size > 3:
        step = float(np.min(np.diff(vv)))
        sigma = g / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if step > 0 and sigma >= _SMEAR_INTERP_STEPS * step:
            S0 = _eval_physical_sharp(vv, ph)
            out = np.zeros_like(S0)
            for dk, wk in zip(d, w):
                out += wk * np.interp(vv - dk, vv, S0, left=0.0, right=0.0)
            return out
    out = np.zeros(np.shape(vv), dtype=float)
    for dk, wk in zip(d, w):
        out += wk * _eval_physical_sharp(v, ph, dshift=dk)
    return out


def _physical_table(INS):
    """(centre, grid, S_unit_area, upper-tail integral) for a KIND_PHYSICAL INS.

    Cached per parameter set: an ordinary spectrum fit keeps one instrumental
    function, so the dynamical diffraction is computed once per worker process
    and every later transmission-integral node is an interpolation.

    The grid is built in two stages -- a coarse scan to locate the line, then a
    dense core of +-12 FWHM around it with log-spaced E^-4 wings -- because a
    grid fine enough for the line everywhere would be four times the cost, and
    the NEW instrumental-function search rebuilds this table on every
    Levenberg-Marquardt step.
    """
    # EXACT key, never rounded. A least-squares Jacobian perturbs a parameter by
    # |p|/1e6 (minimi_lib.compute_jacobian), which for a quantity of order 0.2 --
    # dEQ, say -- is 2e-7. Round the key anywhere near that and the perturbed
    # call returns the cached UNPERTURBED shape, the derivative comes out exactly
    # zero, and the optimiser quietly deactivates the parameter: it stays at its
    # starting value and nothing reports a problem.
    # SHIFT IS FACTORED OUT OF THE KEY. It only TRANSLATES the source -- checked
    # exactly against a recomputed shape -- so the dynamical calculation is done
    # once at shift = 0 and the grid and centre are moved afterwards. Without
    # this every distinct velocity zero cost a full rebuild (115 ms), which made
    # shift the most expensive parameter in the search despite being the
    # cheapest one physically: it is free in every pass, in the Jacobian column
    # that perturbs it, and in any scan over it.
    arr = np.asarray(INS, dtype=float)
    shift = float(arr[_SHIFT_INDEX])
    base = arr.copy()
    base[_SHIFT_INDEX] = 0.0
    key = tuple(base.tolist())
    hit = _SHAPE_CACHE.get(key)
    if hit is not None:
        if shift == 0.0:
            return hit
        c0, g0, S0, tail0 = hit
        return (c0 + shift, g0 + shift, S0, tail0)

    ph = decode_physical(base)

    # The scan only LOCATES the line so the real grid can be placed around it,
    # so it uses the sharp shape: smearing it would double the cost of building
    # the table for a centre the symmetric kernel does not move anyway. The
    # width does grow, and the core half-width below depends on it, so the smear
    # is added back in quadrature -- exact for a Gaussian and ample here, since
    # the core is SHAPE_CORE_FWHM widths wide to begin with.
    scan = np.linspace(-SHAPE_SCAN_MMPS, SHAPE_SCAN_MMPS, SHAPE_SCAN_PTS)
    w, c, _a = line_metrics(scan, _eval_physical_sharp(scan, ph))
    if not np.isfinite(w) or w <= 0:
        w = 10.0 * NAT_WIDTH
    g_smear = float(ph.get('gauss_fwhm', 0.0))
    if g_smear > 0:
        w = float(np.hypot(w, g_smear))
    if not np.isfinite(c):
        c = 0.0
    half = max(SHAPE_CORE_FWHM * w, 4.0 * NAT_WIDTH)
    half = min(half, SHAPE_EDGE_MMPS * 0.5)

    core = np.linspace(-half, half, SHAPE_CORE_PTS)
    wing = np.logspace(np.log10(half), np.log10(SHAPE_EDGE_MMPS),
                       SHAPE_WING_PTS)[1:]
    u = np.concatenate((-wing[::-1], core, wing))       # offsets from the centre
    v = c + u
    S = _eval_physical(v, ph)

    # area = tabulated part + the two analytic E^-4 tails beyond the grid
    area = float(_trapz(S, v)) + (S[0] + S[-1]) * SHAPE_EDGE_MMPS / 3.0
    if not np.isfinite(area) or area <= 0:
        area = 1.0
    S = S / area

    # upper-tail integral on the grid, plus the analytic tail past the last point
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (S[1:] + S[:-1]) * np.diff(v))))
    tail = (cum[-1] - cum) + S[-1] * SHAPE_EDGE_MMPS / 3.0

    out = (float(c), v, S, tail)
    if len(_SHAPE_CACHE) >= _SHAPE_CACHE_MAX:
        _SHAPE_CACHE.clear()
    _SHAPE_CACHE[key] = out
    if shift == 0.0:
        return out
    return (out[0] + shift, out[1] + shift, out[2], out[3])


def gaussian_sum(v_mmps, INS):
    r"""The legacy empirical instrumental function: a sum of Gaussians.

        S(v) = sum_i a_i^2 * Gauss(v - pos_i; sigma_i),  sigma_i = w_i^2 + G_nat/2

    ``INS`` is the flat (w, pos, a) x n array.  This is the same expression
    models.TImod evaluates inside the transmission integral (there it is written
    in the integration variable, with the 1/MulCo Jacobian folded in); it is
    repeated here so ins_shape / ins_centroid / the plots can treat all three
    instrumental-function forms alike.  The transmission integral itself still
    uses its own copy, untouched.
    """
    p = np.atleast_1d(np.asarray(INS, dtype=float))
    v = np.asarray(v_mmps, dtype=float)
    out = np.zeros(np.shape(v), dtype=float)
    for i in range(len(p) // 3):
        sig = p[3 * i] ** 2 + NAT_WIDTH / 2
        out = out + p[3 * i + 2] ** 2 * np.exp(
            -(v - p[3 * i + 1]) ** 2 / (2 * sig ** 2)) / sig / np.sqrt(2 * np.pi)
    return out


def _erfc_half(x):
    """0.5*erfc(x/sqrt2) = the upper tail of a standard normal, vectorised.

    math.erfc is scalar-only and scipy.special is not guaranteed at runtime, so
    this uses the identity erfc(z) = 2*Phi(-z sqrt2) evaluated through a
    rational approximation accurate to ~1e-7 -- far below what the integration
    limits it feeds ever need.
    """
    z = np.abs(np.asarray(x, dtype=float)) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    erf_val = np.where(np.asarray(x, dtype=float) >= 0, y, -y)
    return 0.5 * (1.0 - erf_val)


def ins_shape(INS, v_mmps):
    """
    The source spectral density S(v) [per mm/s, unit area] held by ``INS``.

    Works for all three forms, so plots and diagnostics need no branch of their
    own.  models.TImod keeps its own copy of the Gaussian sum: that path is
    performance-critical and must stay byte-identical to what it always was.
    """
    kind = ins_kind(INS)
    v = np.asarray(v_mmps, dtype=float)
    if kind == KIND_GAUSS:
        return gaussian_sum(v, INS)
    if kind == KIND_RATIONAL:
        return rational_sum(v, decode_rational(INS))
    if kind == KIND_PHYSICAL:
        c, g, S, _tail = _physical_table(INS)
        out = np.interp(v, g, S)
        # analytic E^-4 continuation outside the tabulated range, about the line
        # centre (that is where the sum rule sum_j c_j = 0 makes chi_H ~ E^-2)
        lo, hi = v < g[0], v > g[-1]
        if np.any(lo):
            out = np.where(lo, S[0] * ((g[0] - c)
                                       / (np.where(lo, v, g[0]) - c)) ** 4, out)
        if np.any(hi):
            out = np.where(hi, S[-1] * ((g[-1] - c)
                                        / (np.where(hi, v, g[-1]) - c)) ** 4, out)
        return out
    raise ValueError(f"ins_shape: unknown instrumental-function kind {kind!r}")


def ins_tail_above(INS, v_mmps):
    """int_v^inf S(t) dt for any ``INS`` (total area 1)."""
    kind = ins_kind(INS)
    v = np.asarray(v_mmps, dtype=float)
    if kind == KIND_GAUSS:
        p = np.atleast_1d(np.asarray(INS, dtype=float))
        out = np.zeros(np.shape(v), dtype=float)
        for i in range(len(p) // 3):
            sig = p[3 * i] ** 2 + NAT_WIDTH / 2
            out = out + p[3 * i + 2] ** 2 * _erfc_half((v - p[3 * i + 1]) / sig)
        return out
    if kind == KIND_RATIONAL:
        return rational_sum_tail(v, decode_rational(INS))
    if kind == KIND_PHYSICAL:
        c, g, S, tail = _physical_table(INS)
        out = np.interp(v, g, tail)
        lo, hi = v < g[0], v > g[-1]
        if np.any(lo):
            out = np.where(lo, 1.0 - S[0] * (g[0] - c) ** 4
                           / (3.0 * np.abs(np.where(lo, v, g[0]) - c) ** 3), out)
        if np.any(hi):
            out = np.where(hi, S[-1] * (g[-1] - c) ** 4
                           / (3.0 * (np.where(hi, v, g[-1]) - c) ** 3), out)
        return out
    raise ValueError(f"ins_tail_above: unknown instrumental-function kind {kind!r}")


def ins_centroid(INS):
    r"""First moment  int v S(v) dv  of a theoretical instrumental function [mm/s].

    This is the quantity the legacy code spells ``sum_i pos_i * amp_i^2`` (for a
    sum of unit-area Gaussians with sum amp^2 = 1 that IS the first moment): the
    apparent displacement a multi-line source gives every absorber line.  It
    shifts the drawn line-position markers (models_positions.pos_ac) and the
    folded velocity axis of a calibration (Calibration.py).

    The E^-4 tails make the first moment convergent (the second is too; the third
    is not -- see rational_moments).
    """
    kind = ins_kind(INS)
    if kind == KIND_GAUSS:
        p = np.atleast_1d(np.asarray(INS, dtype=float))
        return float(sum(p[3 * i + 1] * p[3 * i + 2] ** 2
                         for i in range(len(p) // 3)))
    if kind == KIND_RATIONAL:
        p = decode_rational(INS)
        return float(sum(p[5 * m + 4] ** 2
                         * rational_moments(*p[5 * m:5 * m + 4])[0]
                         for m in range(len(p) // 5)))
    if kind == KIND_PHYSICAL:
        c, v, S, _tail = _physical_table(INS)
        m = float(_trapz(v * S, v))
        # analytic E^-4 tails, about the line centre
        for Se, ue in ((S[0], v[0] - c), (S[-1], v[-1] - c)):
            m += Se * ue ** 4 * (c / (3.0 * abs(ue) ** 3)
                                 + np.sign(ue) / (2.0 * ue ** 2))
        return m
    raise ValueError(f"ins_centroid: unknown instrumental-function kind {kind!r}")


def ins_metrics(INS, v=None):
    """(FWHM, centre, area) in mm/s of an instrumental function.

    WARNING -- both the FWHM and the centre here are HALF-MAXIMUM quantities:
    the distance between the outermost points at half the peak, and their
    midpoint. On a source line that develops a second peak (which this one does,
    over most of its operating range) they jump DISCONTINUOUSLY the moment that
    second peak crosses half maximum. The jump is in the definition, not in the
    physics. Use :func:`ins_moments` for anything that has to vary smoothly with
    the parameters -- a plot against angle or temperature, or a fit.
    """
    if v is None:
        v = np.linspace(-2.0, 2.0, 40001)
    return line_metrics(v, ins_shape(INS, v))


def ins_moments(INS, v=None):
    """(centroid, rms width, full width at half maximum of an equivalent
    Gaussian) in mm/s -- all smooth functions of the parameters.

    The E^-4 tails make the first and second moments converge (the third does
    not), so the centroid and the rms width are well defined, continuous, and do
    not care how many peaks the line has. ``2.3548 * rms`` is quoted as the
    third value because that is the FWHM a Gaussian of the same rms would have,
    which is the number most people have an intuition for.
    """
    if v is None:
        v = np.linspace(-6.0, 6.0, 120001)
    S = ins_shape(INS, v)
    area = float(_trapz(S, v))
    if area <= 0:
        return np.nan, np.nan, np.nan
    c = ins_centroid(INS)
    var = float(_trapz((v - c) ** 2 * S, v)) / area
    rms = np.sqrt(max(var, 0.0))
    return c, rms, 2.3548200450309493 * rms


def reduce_to_gaussians(E, S, n_terms=3, passes=3):
    """
    Reduce a numerically computed S(E) to the LEGACY sum-of-Gaussians form.

    Not an improvement -- the opposite: it is the best empirical stand-in for the
    theoretical shape, so that the conventional ``INSexp.txt`` / ``#@INSexp``
    representation stays consistent with the theoretical one for any reader that
    does not understand the accurate parameters.  The Gaussian tails are wrong by
    construction (exp(-v^2) instead of v^-4), which is exactly why the returned
    rms is worth reporting.

    ``E``/``S`` in mm/s.  Returns ``(params, rms, max_dev)`` with params in the
    stored layout (w, pos, a) x n, where the Gaussian width is w^2 + G_nat/2 and
    the amplitudes satisfy sum a^2 = 1.
    """
    import syncmoss.minimi_lib as mi

    E = np.asarray(E, dtype=float)
    S = np.asarray(S, dtype=float)
    area = float(_trapz(S, E))
    if area <= 0:
        raise ValueError("reduce_to_gaussians: S(E) has non-positive area")
    Sn = S / area
    peak = Sn.max()
    y = Sn / peak
    fwhm, centre, _ = line_metrics(E, Sn)
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = 2.0 * NAT_WIDTH
    if not np.isfinite(centre):
        centre = 0.0
    sig = fwhm / 2.3548200450309493

    def shape(x, pp):
        return gaussian_sum(x, pp) / peak

    p = []
    for i in range(int(n_terms)):
        # widths spanning the line, positions spread across it
        s_i = sig * (0.6 + 0.7 * i)
        w_i = np.sqrt(max(s_i - NAT_WIDTH / 2, 1e-6))
        off = 0.0 if n_terms == 1 else (i / (n_terms - 1.0) - 0.5) * fwhm
        p += [w_i, centre + off, 1.0 / np.sqrt(n_terms)]
    p = np.array(p, dtype=float)

    lo = np.array([-np.inf] * len(p))
    hi = np.array([np.inf] * len(p))
    for i in range(int(n_terms)):
        lo[3 * i], hi[3 * i] = -3.0, 3.0                 # w (width = w^2 + G/2)
        lo[3 * i + 1] = centre - 10.0 * fwhm
        hi[3 * i + 1] = centre + 10.0 * fwhm
    bounds = np.array([lo, hi])

    def rms_of(pp):
        if pp is None or not np.all(np.isfinite(pp)):
            return np.inf
        return float(np.sqrt(np.mean(((gaussian_sum(E, pp) - Sn) / peak) ** 2)))

    best, best_rms = p, rms_of(p)
    for _ in range(max(int(passes), 1)):
        try:
            p = mi.minimi_hi(shape, E, y, p, bounds=bounds,
                             tau0=1e-4, MI=30, MI2=30, eps=1e-12)[0]
        except Exception:
            break
        r = rms_of(p)
        if not np.isfinite(r):
            break
        if r < best_rms:
            best_rms, best = r, np.array(p, dtype=float)

    p = np.array(best, dtype=float)
    sc = float(np.sum(p[2::3] ** 2))
    if sc > 0:
        p[2::3] = np.sqrt(p[2::3] ** 2 / sc)
    dev = (gaussian_sum(E, p) - Sn) / peak
    return p, float(np.sqrt(np.mean(dev ** 2))), float(np.max(np.abs(dev)))


def physical_to_gaussians(INS, n_terms=3, v=None):
    """Best legacy Gaussian-sum stand-in for a theoretical instrumental function.

    Returns ``(params, rms, max_dev)`` relative to the peak.  Fitted on a grid
    that stops at +-3 mm/s: a Gaussian sum cannot follow the v^-4 tails at all, so
    including them would only spoil the core it CAN follow.
    """
    if v is None:
        v = np.linspace(-3.0, 3.0, 6001)
    return reduce_to_gaussians(v, ins_shape(INS, v), n_terms=n_terms)


def physical_to_rational(INS, n_terms=2):
    """Reduce a KIND_PHYSICAL instrumental function to a KIND_RATIONAL one.

    Returns ``(INS_rational, rms, max_dev)`` with the deviations relative to the
    peak -- report them: they are the price of the fast closed-form shape.
    """
    ph = decode_physical(INS)
    p, rms, mx = theory_to_rational(
        ph['theta_urad'], ph['B_s'], ph['dEQ'], ph['shift'], n_terms=n_terms,
        thickness_um=ph['thickness_um'], mosaic_urad=ph['mosaic_urad'],
        setting_urad=ph['setting_urad'], phi_m_deg=ph['phi_m_deg'],
        f_LM=ph['f_LM'], dBs_rel_fwhm=ph['dBs_rel_fwhm'],
        n_angle=int(ph['n_angle']), n_field=int(ph['n_field']),
        N_order=int(ph['N_order']))
    return encode_rational(p), rms, mx


def describe_ins(INS):
    """One-line human-readable description of an instrumental function."""
    kind = ins_kind(INS)
    if kind == KIND_GAUSS:
        n = len(np.atleast_1d(np.asarray(INS, dtype=float))) // 3
        return f"empirical sum of {n} Gaussian(s)"
    if kind == KIND_RATIONAL:
        p = decode_rational(INS)
        w, c, _ = ins_metrics(INS)
        return (f"theoretical SMS, {len(p)//5}-term rational form "
                f"(FWHM {w:.4f} mm/s, centre {c:+.4f} mm/s)")
    ph = decode_physical(INS)
    w, c, _ = ins_metrics(INS)
    return (f"Simulation of 57FeBO3 "
            f"(theta {ph['theta_urad']:.1f} urad, B_s {ph['B_s']:.3f} T, "
            f"dEQ {ph['dEQ']:+.3f} mm/s, shift {ph['shift']:+.4f} mm/s; "
            f"FWHM {w:.4f} mm/s, centre {c:+.4f} mm/s)")


# ======================================================================
# 13.  SELF-TESTS
# ======================================================================

def selftest(verbose=True):
    """Validation suite.  Returns True when every check passes."""
    ok = [True]

    def chk(name, cond, extra=""):
        cond = bool(cond)
        ok[0] = ok[0] and cond
        if verbose:
            print(f"  [{'OK  ' if cond else 'FAIL'}] {name} {extra}")

    print("\n--- constants ---------------------------------------------------")
    from syncmoss.constants import sigma as SIGMA_M2, ALPHA_FE_FIELD, \
        LINE_SHIFT_16, LINE_SHIFT_34, ALPHA_FE_V_OUTER
    chk("sigma_0 derived from alpha_ic matches constants.sigma",
        abs(SIGMA0_A2 * 1e-20 / SIGMA_M2 - 1) < 2e-4,
        f"{SIGMA0_A2*1e-20:.4e} vs {SIGMA_M2:.4e} m^2")
    chk("Gamma_0 == NAT_WIDTH in neV",
        abs(GAMMA0_NEV - NAT_WIDTH * MMPS_TO_NEV) < 1e-12,
        f"{GAMMA0_NEV:.4f} neV")

    print("\n--- angular momentum --------------------------------------------")
    chk("CG <1/2 1/2;1 1|3/2 3/2> = 1",
        abs(clebsch_gordan(.5, .5, 1, 1, 1.5, 1.5) - 1) < 1e-12)
    chk("CG <1/2 -1/2;1 1|3/2 1/2> = 1/sqrt3",
        abs(clebsch_gordan(.5, -.5, 1, 1, 1.5, .5) - 1 / np.sqrt(3)) < 1e-12)
    chk("CG <1/2 1/2;1 0|3/2 1/2> = sqrt(2/3)",
        abs(clebsch_gordan(.5, .5, 1, 0, 1.5, .5) - np.sqrt(2 / 3)) < 1e-12)
    G = np.einsum('iba,jba->ij', _T_LAB.conj(), _T_LAB)
    chk("Tr(t_i^+ t_j) = (4/3) delta_ij",
        np.allclose(G, (4 / 3) * np.eye(3), atol=1e-12))

    print("\n--- 57Fe sextet, against SYNCmoss's own line positions -----------")
    site = HyperfineSite(ALPHA_FE_FIELD, np.array([0., 0., 1.]), 0.0)
    w = np.einsum('iba,iba->ba', site.t.conj(), site.t).real.ravel()
    pos = site.dE.ravel()
    keep = w > 1e-9
    o = np.argsort(pos[keep])
    inten = w[keep][o]
    chk("6 allowed lines out of 8", keep.sum() == 6, f"got {keep.sum()}")
    chk("CG weights 3:2:1:1:2:3",
        np.allclose(inten / inten[2], [3, 2, 1, 1, 2, 3], atol=1e-9),
        f"got {np.round(inten/inten[2], 4)}")
    p = pos[keep][o] / MMPS_TO_NEV
    tgt_out = LINE_SHIFT_16 * ALPHA_FE_FIELD
    tgt_in = LINE_SHIFT_34 * ALPHA_FE_FIELD
    chk("outer line == constants.LINE_SHIFT_16 * H",
        abs(p[-1] - tgt_out) < 1e-9, f"{p[-1]:.6f} vs {tgt_out:.6f} mm/s")
    chk("inner line == constants.LINE_SHIFT_34 * H",
        abs(p[3] - tgt_in) < 1e-9, f"{p[3]:.6f} vs {tgt_in:.6f} mm/s")
    chk("outer line == the alpha-Fe standard 5.3123 mm/s",
        abs(p[-1] - ALPHA_FE_V_OUTER) < 1e-9, f"{p[-1]:.6f} mm/s")

    print("\n--- optical theorem ---------------------------------------------")
    geo = Geometry(np.deg2rad(5.1085), 0.0)
    s0 = HyperfineSite(0.0, geo.m_hat, 0.0)
    f = amplitude_matrix(s0, geo.b0, geo.b0, np.array([0.0]), 1.0, 1.0)
    for i, nm in enumerate(("sigma", "pi")):
        sig = 4 * np.pi / K_VEC_A * f[0, i, i].imag
        chk(f"sigma(E0) = sigma_0, {nm} channel",
            abs(sig - SIGMA0_A2) / SIGMA0_A2 < 1e-10,
            f"{sig:.6e} vs {SIGMA0_A2:.6e} A^2")

    print("\n--- pure-nuclear filter theorem ---------------------------------")
    cry = Crystal()
    chk("Bragg angle of (111)_rh = 5.108 deg",
        abs(np.degrees(cry.theta_B) - 5.108) < 0.01,
        f"{np.degrees(cry.theta_B):.4f} deg")
    E = np.array([0.0, 4.0, -7.0])
    chi0, chiH, chiHb = chi_matrices(cry, geo, 0.5, 9.6, 0.0, E)
    sc = np.max(np.abs(chiH))
    chk("chi_H: pi->pi = 0 (m in scattering plane)",
        np.max(np.abs(chiH[:, 1, 1])) < 1e-12 * sc)
    chk("chi_H: sigma->sigma = 0 (m in scattering plane)",
        np.max(np.abs(chiH[:, 0, 0])) < 1e-12 * sc)
    chk("chi_H: sigma->pi non-zero", np.max(np.abs(chiH[:, 1, 0])) > 0)
    s0c = np.max(np.abs(chi0))
    chk("chi_0 is diagonal (m in scattering plane)",
        max(np.max(np.abs(chi0[:, 0, 1])),
            np.max(np.abs(chi0[:, 1, 0]))) < 1e-12 * s0c)

    geo2 = Geometry(np.deg2rad(5.1085), np.pi / 2)
    _, chiH2, _ = chi_matrices(cry, geo2, 0.5, 9.6, 0.0, E)
    chk("m normal to scattering plane -> sigma->pi = 0",
        np.max(np.abs(chiH2[:, 1, 0])) < 1e-12 * np.max(np.abs(chiH2)))
    chk("m normal to scattering plane -> sigma->sigma survives",
        np.max(np.abs(chiH2[:, 0, 0])) > 0)

    _, chiH0, _ = chi_matrices(cry, geo, 0.0, 9.6, 0.0, E)
    chk("B_s = 0 -> pure-nuclear structure factor vanishes",
        np.max(np.abs(chiH0)) < 1e-24, f"max {np.max(np.abs(chiH0)):.2e}")

    Ew = np.linspace(-400, 400, 16001)
    c0w, cHw, _ = chi_matrices(cry, geo, ALPHA_FE_FIELD, 0.0, 0.0, Ew)

    def poles(y, rel=1e-3):
        i = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
                     & (y[1:-1] > rel * y.max()))[0] + 1
        return Ew[i] / MMPS_TO_NEV, y[i]

    pH, vH = poles(np.abs(cHw[:, 1, 0]))
    chk("chi_H has exactly 4 resonances (dm = 0 cancels)", pH.size == 4,
        f"got {pH.size} at {np.round(pH, 3)} mm/s")
    if pH.size == 4:
        chk("chi_H lines at the alpha-Fe 1,3,4,6 positions",
            np.allclose(np.abs(pH), [tgt_out, tgt_in, tgt_in, tgt_out], atol=0.01),
            f"{np.round(pH, 3)}")
        chk("chi_H weights outer:inner = 3:1",
            abs(vH[0] / vH[1] - 3.0) < 0.05, f"got {vH[0]/vH[1]:.3f}")
    p0, _ = poles(np.abs(c0w[:, 0, 0] - cry.chi0_el))
    chk("chi_0 keeps all 6 lines (no cancellation in forward scattering)",
        p0.size == 6, f"got {p0.size} at {np.round(p0, 3)} mm/s")

    print("\n--- polarization ------------------------------------------------")
    par_p = SMSParams(B_s=ALPHA_FE_FIELD, dEQ=0.0)
    Rp = sms_spectrum(Ew, 0.0, cry, par_p, smear=False, return_amplitude=True)
    j = int(np.argmax(np.abs(Rp[:, 1, 0])))
    pur = abs(Rp[j, 1, 0]) ** 2 / (abs(Rp[j, 1, 0]) ** 2 + abs(Rp[j, 0, 0]) ** 2)
    chk("reflected beam is pure pi (sigma incidence)", pur > 1 - 1e-12,
        f"purity {100*pur:.9f} %")
    chk("peak reflectivity <= 1", abs(Rp[j, 1, 0]) ** 2 <= 1.0,
        f"|R|^2 = {abs(Rp[j,1,0])**2:.4f}")

    print("\n--- dynamical solver --------------------------------------------")
    # sigma->pi and pi->sigma decouple, so each block is a scalar two-beam
    # problem with the classical (Darwin-Prins) semi-infinite solution
    #     s+- = [alpha +- sqrt((alpha - 2 chi_0)^2 - 4 chi_H chi_Hbar)]/2
    #     keep Im s > 0;  R = (s - chi_0)/chi_Hbar
    chi00 = -8.0e-6 + 4.0e-5j
    c0 = np.array([[chi00, 0], [0, chi00]], dtype=complex)[None]
    a_, b_ = 2.0e-5 + 0.8e-5j, 1.4e-5 + 0.5e-5j
    cH = np.array([[0, b_], [a_, 0]], dtype=complex)[None]
    cHb = np.array([[0, a_], [b_, 0]], dtype=complex)[None]
    tb = np.deg2rad(5.1085)

    def darwin_prins(alpha, chi0s, chiHs, chiHbs):
        disc = np.sqrt((alpha - 2 * chi0s) ** 2 - 4 * chiHs * chiHbs)
        s = np.array([(alpha + disc) / 2, (alpha - disc) / 2])
        s = s[np.argmax(s.imag)]
        return (s - chi0s) / chiHbs

    for al in (-6e-5, -2e-5, 0.0, 3e-5, 1e-4):
        R = reflectivity_matrix(c0, cH, cHb, al, 1.0e5, tb)      # 10 cm ~ infinite
        ana = darwin_prins(al, chi00, a_, a_)
        chk(f"Darwin-Prins, alpha={al:+.1e}",
            abs(R[0, 1, 0] - ana) / abs(ana) < 1e-6,
            f"|R|={abs(R[0,1,0]):.6f} vs {abs(ana):.6f}")
        chk(f"    |R| <= 1 at alpha={al:+.1e}", abs(R[0, 1, 0]) <= 1.0 + 1e-9)

    dthin = 2.0e-5
    R = reflectivity_matrix(c0, cH, cHb, 5e-5, dthin, tb)
    kin = 1j * np.pi * (dthin * 1e4) * a_ / (LAMBDA_A * np.sin(tb))
    chk("thin-crystal limit -> kinematical",
        abs(R[0, 1, 0] - kin) / abs(kin) < 1e-3, f"{R[0,1,0]:.4e} vs {kin:.4e}")

    thick = abs(reflectivity_matrix(c0, cH, cHb, 0.0, 1.0e5, tb)[0, 1, 0])
    seq = [abs(reflectivity_matrix(c0, cH, cHb, 0.0, d, tb)[0, 1, 0])
           for d in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    chk("finite d -> semi-infinite limit",
        abs(seq[-1] - thick) / thick < 1e-6 and all(np.diff(seq) > 0),
        f"{np.round(seq, 5)} -> {thick:.5f}")

    # batched alpha must reproduce the one-at-a-time calls exactly
    als = np.array([-6e-5, -2e-5, 0.0, 3e-5])
    Rb = reflectivity_matrix(np.repeat(c0, 4, axis=0), np.repeat(cH, 4, axis=0),
                             np.repeat(cHb, 4, axis=0), als, 35.0, tb)
    Rs = np.array([reflectivity_matrix(c0, cH, cHb, a, 35.0, tb)[0]
                   for a in als])
    chk("batched alpha == scalar alpha (mosaic average is exact)",
        np.max(np.abs(Rb - Rs)) < 1e-15, f"max diff {np.max(np.abs(Rb-Rs)):.1e}")

    print("\n--- kinematical line shape --------------------------------------")
    cry_thin = Crystal(thickness=1.0e-3)
    par = SMSParams(B_s=0.02, dEQ=0.0, mosaic_fwhm_urad=0.0,
                    setting_fwhm_urad=0.0, n_angle=1)
    Eg = np.linspace(-15, 15, 4001)
    S = sms_spectrum(Eg, 3000.0, cry_thin, par, smear=False)
    wdt = line_metrics(Eg, S)[0]
    target = GAMMA0_NEV * np.sqrt(np.sqrt(2) - 1)
    chk("squared-Lorentzian width 0.644 Gamma_0",
        abs(wdt - target) / target < 0.02,
        f"{wdt/GAMMA0_NEV:.3f} vs {target/GAMMA0_NEV:.3f} G0")

    print("\n--- analytic structure of S(E) ----------------------------------")
    for Bs, dq in ((0.5, -0.2 * MMPS_TO_NEV), (ALPHA_FE_FIELD, 0.0),
                   (2.0, 0.4 * MMPS_TO_NEV)):
        cH_, zH = pole_residues(cry, geo, Bs, dq, which="H")
        c0_, z0 = pole_residues(cry, geo, Bs, dq, which="0")
        chk(f"sum c_j = 0 for chi_H (B_s={Bs} T, dEQ={dq/MMPS_TO_NEV:+.2f})",
            abs(cH_.sum()) < 1e-12 * np.max(np.abs(cH_)),
            f"|sum c|/max|c| = {abs(cH_.sum())/np.max(np.abs(cH_)):.1e}, n={cH_.size}")
        chk("    ... but sum d_j != 0 for the forward channel",
            abs(c0_.sum()) > 0.1 * np.max(np.abs(c0_)))
        chk("    chi_H keeps exactly 4 poles (the dm = +-1 lines)",
            cH_.size == 4, f"got {cH_.size}")

    Etail = np.concatenate([-np.logspace(4, 1.5, 200),
                            np.linspace(-30, 30, 300),
                            np.logspace(1.5, 4, 200)])
    for nm, c_, Bs, th in (("near T_N, 35 um, theta=70", Crystal(), 0.5, 70.0),
                           ("thin crystal 0.1 um", Crystal(thickness=0.1), 0.5, 70.0),
                           ("fully split, theta=0", Crystal(), ALPHA_FE_FIELD, 0.0)):
        par_t = SMSParams(B_s=Bs, mosaic_fwhm_urad=0, setting_fwhm_urad=0, n_angle=1)
        St = sms_spectrum(Etail, th, c_, par_t, smear=False)
        pp, pm = tail_exponent(Etail, St)
        chk(f"tail exponent -4: {nm}", abs(pp + 4) < 0.1 and abs(pm + 4) < 0.1,
            f"{pp:+.3f} / {pm:+.3f}")

    # the tail exponent must survive the incoherent average as well
    par_sm = SMSParams(B_s=0.5, dBs_rel_fwhm=0.05, n_field=5)
    Ssm = sms_spectrum(Etail, 70.0, cry, par_sm)
    pp, pm = tail_exponent(Etail, Ssm)
    chk("tail exponent -4 after mosaic + field averaging",
        abs(pp + 4) < 0.1 and abs(pm + 4) < 0.1, f"{pp:+.3f} / {pm:+.3f}")

    print("\n--- width sum rule ----------------------------------------------")
    for al in (-1e-4, -3e-5, -1e-5):
        r0 = dressed_poles(cry, geo, 0.5, -0.2 * MMPS_TO_NEV, al, lossless=True)
        r1 = dressed_poles(cry, geo, 0.5, -0.2 * MMPS_TO_NEV, al, lossless=False)
        n = r0.size
        s0_, s1_ = -r0.imag.sum(), -r1.imag.sum()
        chk(f"sum g_k = n*G0/2 (lossless), alpha={al:+.0e}",
            abs(s0_ / (n * GAMMA0_NEV / 2) - 1) < 1e-9,
            f"ratio {s0_/(n*GAMMA0_NEV/2):.9f}")
        chk(f"    loss pushes the sum UP, alpha={al:+.0e}", s1_ > s0_,
            f"ratio {s1_/(n*GAMMA0_NEV/2):.6f}")

    print("\n--- rational primitive ------------------------------------------")
    v1, g1, v2, g2 = 4.8, 2.9, 23.0, 9.1
    Eg2 = np.linspace(-4.0e4, 4.0e4, 4_000_001)
    Sr = rational_if(Eg2, v1, g1, v2, g2)
    area = float(_trapz(Sr, Eg2))
    cen = float(_trapz(Eg2 * Sr, Eg2) / area)
    cen_f, var_f = rational_moments(v1, g1, v2, g2)
    chk("rational form has unit area", abs(area - 1) < 1e-4, f"{area:.8f}")
    chk("closed-form centroid", abs(cen - cen_f) < 1e-4 * abs(cen_f),
        f"{cen:.6f} vs {cen_f:.6f}")
    var = float(_trapz((Eg2 - cen_f) ** 2 * Sr, Eg2))
    chk("closed-form variance", abs(var / var_f - 1) < 5e-3,
        f"{var:.4f} vs {var_f:.4f}")

    # tail integral must be the exact antiderivative
    Ec = np.linspace(-3.0e4, 3.0e4, 3_000_001)
    Sc = rational_if(Ec, v1, g1, v2, g2)
    for Ex in (-40.0, -5.0, 0.0, 7.0, 55.0):
        num = float(_trapz(Sc[Ec >= Ex], Ec[Ec >= Ex]))
        ana = float(rational_tail_integral(Ex, v1, g1, v2, g2))
        chk(f"closed-form tail integral at E={Ex:+.0f}", abs(num - ana) < 2e-4,
            f"{ana:.6f} vs {num:.6f}")
    chk("tail integral -> 1 at -inf",
        abs(float(rational_tail_integral(-1e7, v1, g1, v2, g2)) - 1) < 1e-4)
    chk("tail integral -> 0 at +inf",
        abs(float(rational_tail_integral(1e7, v1, g1, v2, g2))) < 1e-4)
    # degenerate poles (the exact squared Lorentzian): its own closed form, and
    # it must join continuously onto the general one as the poles separate.
    chk("tail integral at the symmetry point of a squared Lorentzian",
        abs(float(rational_tail_integral(3.0, 3.0, 2.0, 3.0, 2.0)) - 0.5) < 1e-14,
        f"{float(rational_tail_integral(3.0, 3.0, 2.0, 3.0, 2.0)):.15f}")
    Ed = np.linspace(-6000.0, 6000.0, 6_000_001)
    Sd = rational_if(Ed, 3.0, 2.0, 3.0, 2.0)
    for Ex in (-4.0, 0.0, 3.0, 9.0):
        chk(f"    degenerate branch vs quadrature at E={Ex:+.0f}",
            abs(float(rational_tail_integral(Ex, 3.0, 2.0, 3.0, 2.0))
                - float(_trapz(Sd[Ed >= Ex], Ed[Ed >= Ex]))) < 2e-4)
    for eps_ in (1e-3, 1e-5, 1e-7):
        gen = float(rational_tail_integral(1.0, 3.0, 2.0, 3.0, 2.0 * (1 + eps_)))
        chk(f"    general branch -> degenerate one as g2/g1-1 = {eps_:.0e}",
            abs(gen - float(rational_tail_integral(1.0, 3.0, 2.0, 3.0, 2.0)))
            < 10.0 * eps_ + 1e-9, f"{gen:.12f}")

    wq = GAMMA0_NEV / 2
    Eq = np.linspace(-200, 200, 400001)
    Sq = rational_if(Eq, 0.0, wq, 0.0, wq)
    iq = np.where(Sq >= Sq.max() / 2)[0]
    chk("degenerate limit = squared Lorentzian, 0.6436 G0",
        abs((Eq[iq[-1]] - Eq[iq[0]]) / GAMMA0_NEV - 0.6436) < 2e-3,
        f"{(Eq[iq[-1]]-Eq[iq[0]])/GAMMA0_NEV:.4f} G0")

    print("\n--- reduction of the exact S(E) ---------------------------------")
    Ef = np.concatenate([-np.logspace(np.log10(2000), np.log10(60), 150),
                         np.linspace(-60, 60, 1601),
                         np.logspace(np.log10(60), np.log10(2000), 150)])
    for th, Bs in ((30.0, 0.5), (70.0, 0.5), (130.0, 0.5)):
        Sf = sms_spectrum(Ef, th, cry, SMSParams(B_s=Bs))
        prev = np.inf
        for nt, tol in ((1, 0.05), (2, 0.025), (3, 0.02)):
            pr, rms, mx = reduce_to_rational(Ef, Sf, n_terms=nt)
            chk(f"{nt}-term rational reduction at theta={th:.0f}, B_s={Bs}",
                rms < tol, f"rms {100*rms:.2f} %, max {100*mx:.2f} %")
            chk("    more terms never fit worse", rms <= prev + 1e-9,
                f"{100*rms:.2f} % vs {100*prev:.2f} %")
            prev = rms
            # unit area is exact by construction (each term has unit area and the
            # weights are normalised); check that, plus the surviving E^-4 tails.
            chk("    reduced form has unit area analytically",
                abs(np.sum(pr[4::5] ** 2) - 1) < 1e-12,
                f"sum w^2 = {np.sum(pr[4::5]**2):.12f}")
            chk("    ... and numerically, from the closed-form tail integrals",
                abs(float(rational_sum_tail(-1e9, pr))
                    - float(rational_sum_tail(1e9, pr)) - 1) < 1e-6)
            tp, tm = tail_exponent(Ef, rational_sum(Ef, pr))
            chk("    reduced form still has E^-4 tails",
                abs(tp + 4) < 0.05 and abs(tm + 4) < 0.05,
                f"{tp:+.3f} / {tm:+.3f}")

    print("\n--- INS encoding ------------------------------------------------")
    legacy = np.array([0.449, -0.044, 0.449, -0.187, 0.066, 0.757,
                       -0.168, -0.110, 0.475])
    chk("a legacy Gaussian INS is recognised as legacy",
        ins_kind(legacy) == KIND_GAUSS)
    chk("an empty/short INS does not crash the kind test",
        ins_kind([]) == KIND_GAUSS and ins_kind([0.3]) == KIND_GAUSS)
    ph = encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2, shift=0.013)
    chk("physical INS round-trips", ins_kind(ph) == KIND_PHYSICAL
        and abs(decode_physical(ph)['shift'] - 0.013) < 1e-15
        and abs(decode_physical(ph)['theta_urad'] - 70.0) < 1e-15)
    pr = encode_rational([0.0, 0.05, 0.02, 0.09, 1.0])
    chk("rational INS round-trips", ins_kind(pr) == KIND_RATIONAL
        and np.allclose(decode_rational(pr), [0.0, 0.05, 0.02, 0.09, 1.0]))

    vv = np.linspace(-1.5, 1.5, 30001)
    for nm, ins in (("physical", ph), ("rational", pr)):
        Sv = ins_shape(ins, vv)
        ar = float(_trapz(Sv, vv)) + float(ins_tail_above(ins, vv[-1])) \
            + 1.0 - float(ins_tail_above(ins, vv[0]))
        chk(f"{nm} INS has unit area (tabulated part + analytic tails)",
            abs(ar - 1) < 2e-3, f"{ar:.6f}")
        chk(f"{nm} INS: tail integral is the primitive of the shape",
            abs((float(ins_tail_above(ins, -0.3)) - float(ins_tail_above(ins, 0.4)))
                - float(_trapz(Sv[(vv >= -0.3) & (vv <= 0.4)],
                               vv[(vv >= -0.3) & (vv <= 0.4)]))) < 1e-4)
        chk(f"{nm} INS: tail -> 1 below and 0 above",
            abs(float(ins_tail_above(ins, -1e4)) - 1) < 1e-6
            and abs(float(ins_tail_above(ins, 1e4))) < 1e-6)
        chk(f"{nm} INS: shape is non-negative everywhere", np.all(Sv >= 0))
    cph = _physical_table(ph)[0]
    chk("physical shape keeps the E^-4 law outside the tabulated grid",
        abs(np.log(float(ins_shape(ph, cph + 200.0))
                   / float(ins_shape(ph, cph + 400.0))) / np.log(2.0) - 4.0) < 1e-9)
    # the tabulation must not distort the shape: compare against direct evaluation
    vchk = cph + np.array([-0.31, -0.117, -0.041, 0.0, 0.023, 0.067, 0.19, 0.55])
    direct = _eval_physical(vchk, decode_physical(ph))
    direct = direct / (_trapz(_eval_physical(_physical_table(ph)[1],
                                             decode_physical(ph)),
                              _physical_table(ph)[1])
                       + 0.0)          # same normalisation as the table, tails aside
    interp = ins_shape(ph, vchk)
    rel = np.max(np.abs(interp - direct)) / direct.max()
    chk("tabulated physical shape == direct evaluation", rel < 3e-3,
        f"max {100*rel:.4f} % of peak")

    # the legacy Gaussian sum on the same shape: it is the fallback written to
    # INSexp.txt, and its error is the price of not understanding the accurate form
    pg, grms, gmx = physical_to_gaussians(ph, n_terms=3)
    chk("legacy Gaussian stand-in fits the theoretical core", grms < 0.05,
        f"3 Gaussians: rms {100*grms:.2f} %, max {100*gmx:.2f} %")
    chk("    ... and it is a proper legacy INS (sum a^2 = 1, recognised as legacy)",
        ins_kind(pg) == KIND_GAUSS and abs(np.sum(pg[2::3] ** 2) - 1) < 1e-12)
    gt, _ = tail_exponent(np.linspace(-300, 300, 20001),
                          gaussian_sum(np.linspace(-300, 300, 20001), pg),
                          lo=2.0, hi=6.0)
    chk("    ... but its tails are Gaussian, not E^-4 (that is the point)",
        not np.isfinite(gt) or gt < -8.0, f"log-log slope {gt:+.2f} vs -4")

    pr2, rrms, rmx = physical_to_rational(ph, n_terms=2)
    dv = (ins_shape(pr2, vv) - ins_shape(ph, vv)) / ins_shape(ph, vv).max()
    chk("rational reduction of a physical INS agrees with it",
        float(np.sqrt(np.mean(dv ** 2))) < 0.02,
        f"rms {100*np.sqrt(np.mean(dv**2)):.2f} %, reduction rms {100*rrms:.2f} %")
    w1, c1, _ = ins_metrics(ph)
    w2, c2, _ = ins_metrics(pr2)
    chk("    ... and reproduces its FWHM and centre",
        abs(w1 - w2) < 0.02 * w1 and abs(c1 - c2) < 0.01 * max(w1, 1e-9),
        f"FWHM {w1:.4f} -> {w2:.4f} mm/s, centre {c1:+.5f} -> {c2:+.5f} mm/s")

    print("\n--- absorber-area invariance ------------------------------------")
    ta = 2.5
    Ea = np.linspace(-1500, 1500, 60001)
    va = np.linspace(-1200, 1200, 24001)
    for nm, Sc_ in (("exact S(E)", sms_spectrum(Ea, 70.0, cry, SMSParams(B_s=0.50))),
                    ("squared Lorentzian",
                     rational_if(Ea, 0, GAMMA0_NEV / 2, 0, GAMMA0_NEV / 2)),
                    ("Gaussian", np.exp(-0.5 * (Ea / GAMMA0_NEV) ** 2))):
        Sc_ = Sc_ / _trapz(Sc_, Ea)
        Tv = np.array([_trapz(Sc_ * np.exp(
            -ta * (GAMMA0_NEV / 2) ** 2 /
            ((Ea - vv) ** 2 + (GAMMA0_NEV / 2) ** 2)), Ea) for vv in va])
        A = float(_trapz(1 - Tv, va))
        chk(f"area independent of source shape: {nm}",
            abs(A / absorption_area(ta) - 1) < 5e-3,
            f"A/A_delta = {A/absorption_area(ta):.5f}")

    print("\n--- imperfections -----------------------------------------------")
    # the ideal limit the two imperfections exist to spoil
    f_sq, n_sq = squared_lorentzian_limit()
    chk("squared_lorentzian_limit = sqrt(sqrt(2)-1), tails E^-4",
        abs(f_sq - np.sqrt(np.sqrt(2) - 1)) < 1e-12 and n_sq == -4,
        f"{f_sq:.6f} Gamma_0, exponent {n_sq}")

    base = encode_physical(theta_urad=70.0, B_s=0.50, dEQ=-0.3228)
    c0, r0, _g0 = ins_moments(base)
    prev = r0
    for gf in (0.05, 0.15, 0.30):
        INS = encode_physical(theta_urad=70.0, B_s=0.50, dEQ=-0.3228,
                              gauss_fwhm=gf)
        c1, r1, _g1 = ins_moments(INS)
        sig = gf / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        want = np.sqrt(r0 ** 2 + sig ** 2)
        chk(f"gauss_fwhm={gf}: rms widths add in quadrature",
            r1 > prev and abs(r1 - want) < 0.05 * want,
            f"rms {r0:.4f} -> {r1:.4f}, quadrature {want:.4f} mm/s")
        chk(f"gauss_fwhm={gf}: centroid and area untouched",
            abs(c1 - c0) < 1e-3 * max(abs(c0), 1e-3) + 1e-6,
            f"centroid {c0:+.5f} -> {c1:+.5f} mm/s")
        vt = np.array([8.0, 16.0])
        St = ins_shape(INS, vt)
        slope = float(np.log(St[1] / St[0]) / np.log(vt[1] / vt[0]))
        chk(f"gauss_fwhm={gf}: E^-4 wings survive the smear",
            abs(slope + 4.0) < 0.2, f"log-log slope {slope:+.2f}")
        prev = r1

    # dT_mK: ONE thermal spread gives a field spread that scales as 1/(T - T_N),
    # which is the whole reason it is a better shared parameter than dBs_rel.
    rel = []
    for B in (2.30, 0.58, 0.18):
        dB, w = _field_nodes(SMSParams(B_s=B, dT_mK=5.0, n_field=9))
        rel.append(float(np.sqrt(np.sum(w * dB ** 2)) / B))
    chk("dT_mK: the relative field spread grows as T -> T_N",
        rel[0] > rel[1] > rel[2] and rel[0] / rel[2] > 3.0,
        "  ".join(f"{100*r:.2f} %" for r in rel))
    dB, w = _field_nodes(SMSParams(B_s=0.58, dT_mK=5.0, n_field=9))
    chk("dT_mK: the field average is still centred on B_s",
        abs(float(np.sum(w * dB))) < 1e-3 * 0.58,
        f"mean offset {float(np.sum(w*dB)):+.2e} T")
    dB0, _w0 = _field_nodes(SMSParams(B_s=0.58, dT_mK=0.0, dBs_rel_fwhm=0.0,
                                      n_field=9))
    chk("no imperfection -> a single field node",
        dB0.size == 1 and dB0[0] == 0.0, f"{dB0.size} node(s)")

    print("\n  SMS THEORY SELFTEST:",
          "ALL PASSED" if ok[0] else "FAILURES PRESENT", "\n")
    return ok[0]


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
