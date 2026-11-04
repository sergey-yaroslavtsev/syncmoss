"""The mosaic textured full-Hamiltonian component ('Hamiltonian').

Two things are pinned here.

(1) THE CONVENTION. The Hamiltonian family used to build its per-transition 2x2
    cross-section blocks CONJUGATED (equivalently: transposed) -- the amplitudes
    were formed as Vex * conj(Vgr) instead of conj(Vex) * Vgr, and the mirrored
    azimuth of the polarization components compensated that exactly in every
    scalar intensity |A|^2, so nothing was visibly wrong. It is visible in the
    2x2 block: the Delta m = +1 (sigma+) lines carried the sigma- Faraday
    (magneto-optical) sign of the Sextet family, so a Hamiltonian component
    mixed or stacked with any other polarized component had the WRONG relative
    polarity. ``_textbook_blocks`` below rebuilds the blocks from first
    principles (explicit spin operators, standard H_Q/H_Z, Clebsch--Gordan M1
    dipoles) and the tests check the code against it, and against the Sextet
    building blocks in the pure-Zeeman limit -- including a two-layer stack, the
    geometry in which the sign is a first-order observable.

(2) THE ORDER PARAMETERS. 'Hamiltonian' averages the crystal orientation over an
    axially symmetric ODF, in closed form, through three order parameters
    (A, Am, Ah). ``_brute_force_blocks`` performs the same average by explicit
    numerical integration over the ODF and the closed form must reproduce it, for
    two different realisations of the tilt distribution. The exact limits
    (1,1,1) = single crystal, (0,0,0) = random powder, Ah = 0 = fiber texture
    are checked against the models they replace (Hamilton_mc / Hamilton_pc /
    the textured Sextet).

(3) THE CLOSED FORM. The family no longer diagonalises (Szymanski 2026, see
    docs/Hamiltonian_analytical.md); section (6) checks its lines against first
    principles and against the *_classic eigensolver versions -- also where two
    excited levels coincide, which the paper's formulas cannot do -- and that it
    stays smooth through those points.

Pure NumPy/Numba path -- no Qt, no multiprocessing.
"""
import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.constants import c, E0_J, ggr, gex, mun, NAT_WIDTH

_E = np.linspace(-11.0, 11.0, 160)

# T, delta, Q, H, L, G, eta, thetaH, phiH, theta, phi, alpha_k  (12 = Hamilton_mc)
_HAM12 = [8.0, 0.1, 0.4, 33.0, 0.098, 0.12, 0.3, 20.0, 30.0, 40.0, 50.0, 70.0]
# the same at Q = eta = delta = 0 -> a pure Zeeman sextet
_TET, _PHI = 41.0, 17.0
_GEO = dict(tetr=54.0, phir=71.0, alfak=23.0)
_HAMQ0 = [8.0, 0.0, 0.0, 33.0, 0.098, 0.12, 0.0, _TET, _PHI,
          _GEO['tetr'], _GEO['phir'], _GEO['alfak']]


def _run(models, params, mett=0, pol=0.98):
    """Per-energy absorber transmission for a component list."""
    m5.COMPLEX_VOIGT_METHOD = 'pseudo'
    m5.DISPERSION_SIGN = +1.0
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array(list(models)), _E, 0.0, 1.0,
                 np.array([]), [], [], Met=-1, Mett=mett, sms_pol=pol),
        dtype=float,
    )


# --------------------------------------------------------------------------
# an independent, textbook reference for the per-transition 2x2 blocks
# --------------------------------------------------------------------------
def _spin_ops(I):
    n = int(round(2 * I + 1))
    m = np.array([I - k for k in range(n)])
    Iz = np.diag(m).astype(complex)
    Ip = np.zeros((n, n), complex)
    for k in range(1, n):                              # <m+1|I+|m>
        Ip[k - 1, k] = np.sqrt(I * (I + 1) - m[k] * (m[k] + 1))
    Im = Ip.conj().T
    return (Ip + Im) / 2, (Ip - Im) / 2j, Iz


def _sph(theta_deg, phi_deg):
    t, f = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])


# <3/2 m_e|T_q|1/2 m_g> = CG(1/2 m_g; 1 q|3/2 m_e); rows m_e = 3/2..-3/2, cols m_g = 1/2,-1/2
_TQ = {}
_TQ[+1] = np.zeros((4, 2)); _TQ[+1][0, 0] = 1.0;            _TQ[+1][1, 1] = np.sqrt(1 / 3)
_TQ[0] = np.zeros((4, 2));  _TQ[0][1, 0] = np.sqrt(2 / 3);  _TQ[0][2, 1] = np.sqrt(2 / 3)
_TQ[-1] = np.zeros((4, 2)); _TQ[-1][2, 0] = np.sqrt(1 / 3); _TQ[-1][3, 1] = 1.0


def _dipoles(Q, Hhf, eta, tet, phi):
    """(D, S): complex dipole 3-vectors in EFG coords and positions, per transition.

    A(e) = D . e is the M1 transition amplitude for a (real) polarization e:
    A = sum_q (-1)^q e_{-q} <e|T_q|g>, the standard right-handed convention.
    The cross-section block is then P_ab = 3/4 * (D.e_a) conj(D.e_b), the 3/4
    fixing the same normalisation the code uses (checked by the powder sum rule).
    """
    Ix, Iy, Iz = _spin_ops(1.5)
    nB = _sph(tet, phi)
    Hex_m = (Q / c * E0_J / 12.0 * (3 * Iz @ Iz - 1.5 * 2.5 * np.eye(4)
                                    + eta * (Ix @ Ix - Iy @ Iy))
             - Hhf * gex * mun * (nB[0] * Ix + nB[1] * Iy + nB[2] * Iz))
    gx, gy, gz = _spin_ops(0.5)
    Hgr_m = -Hhf * ggr * mun * (nB[0] * gx + nB[1] * gy + nB[2] * gz)
    # Hermitian solver: orthonormal eigenvectors also inside a degenerate level
    Hex, Vex = np.linalg.eigh(Hex_m)
    Hgr, Vgr = np.linalg.eigh(Hgr_m)
    Hex, Hgr = np.real(Hex), np.real(Hgr)
    axes = []
    for u in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        e_p1, e_m1 = -(u[0] + 1j * u[1]) / np.sqrt(2), (u[0] - 1j * u[1]) / np.sqrt(2)
        axes.append(Vex.conj().T @ (u[2] * _TQ[0] - e_m1 * _TQ[+1] - e_p1 * _TQ[-1]) @ Vgr)
    D = np.zeros((8, 3), complex)
    S = np.zeros(8)
    for k in range(8):
        i, j = k % 4, k // 4
        D[k] = [axes[0][i, j], axes[1][i, j], axes[2][i, j]]
        S[k] = (Hex[i] - Hgr[j]) / E0_J * c
    return D, S


def _lab_axes(tetr, phir, alfak, cms=False):
    """Lab axes (e1, e2, k) as rows, in EFG coords -- the geometry TImod uses."""
    t, f, a = np.radians(tetr), np.radians(phir), np.radians(alfak)
    nr = _sph(tetr, phir)
    that = np.array([np.cos(t) * np.cos(f), np.cos(t) * np.sin(f), -np.sin(t)])
    phat = np.array([-np.sin(f), np.cos(f), 0.0])
    if cms:                                   # (tetr, phir) is the BEAM k
        return np.vstack([that, phat, nr])
    return np.vstack([nr, np.sin(a) * that - np.cos(a) * phat,
                      np.cos(a) * that + np.sin(a) * phat])


def _textbook_blocks(Q, Hhf, eta, tet, phi, R):
    D, S = _dipoles(Q, Hhf, eta, tet, phi)
    dl = D @ R.T
    P = np.array([0.75 * np.outer(dl[k, :2], dl[k, :2].conj()) for k in range(8)])
    return P, S


def _rot(axis, ang):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _brute_force_blocks(Q, Hhf, eta, tet, phi, R0, fl, Atex, Am, Ah,
                        nbeta=180, nalpha=1001, generic=False):
    """The SAME orientation average, done by explicit numerical integration.

    ODF: R = W(beta, chi) R0 Z(alpha) -- Z a rotation of the crystal about the
    reference axis with <cos m alpha> = Ah^|m| (a wrapped Cauchy, exactly), and W
    the twist-free wobble about the lab reference axis ``fl`` (beta uniform, chi
    from a 3-point distribution with the prescribed <cos chi>, <cos^2 chi>). The
    two stages are independent, so <R_ai R_bj> factorises over them.
    """
    c2 = max(0.0, (1.0 + 2.0 * Atex) / 3.0)
    c1 = Am * np.sqrt(c2)
    if generic:
        # A DIFFERENT realisation: three GENERIC (non-special) tilt angles rather
        # than 0/90/180 deg. Only the first two moments of cos(chi) may matter, so
        # any support that can carry them must give the same answer; pick the
        # first candidate whose weights come out non-negative.
        for sup in [(0.97, 0.31, -0.93), (0.88, -0.07, -0.99), (0.73, 0.11, -0.66),
                    (0.51, -0.03, -0.44), (0.35, 0.02, -0.29), (0.19, 0.01, -0.16)]:
            u = np.array(sup)
            w = np.linalg.solve(np.vstack([np.ones(3), u, u ** 2]), np.array([1.0, c1, c2]))
            if np.all(w > 1e-6):
                break
        else:
            pytest.skip("no generic 3-point tilt support carries these moments")
    else:                                      # tilts of 0, 90, 180 degrees
        u = np.array([1.0, 0.0, -1.0])
        w = np.array([(c2 + c1) / 2, 1.0 - c2, (c2 - c1) / 2])
    assert np.all(w > -1e-12), "unrealizable moments for this support"
    al = np.linspace(-np.pi, np.pi, nalpha, endpoint=False)
    if Ah >= 1.0:
        al, wa = np.zeros(1), np.ones(1)
    elif Ah == 0.0:
        wa = np.full(nalpha, 1.0 / nalpha)
    else:
        d = (1 - Ah ** 2) / (1 + Ah ** 2 - 2 * Ah * np.cos(al))
        wa = d / d.sum()
    n0 = R0.T @ fl
    perp = np.array([0.0, 0.0, 1.0]) if abs(fl[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp = perp - fl * (perp @ fl)
    perp /= np.linalg.norm(perp)
    Wm, Ww = [], []
    for ui, wi in zip(u, w):
        chi = np.arccos(np.clip(ui, -1, 1))
        for b in np.linspace(0, 2 * np.pi, nbeta, endpoint=False):
            Wm.append(_rot(_rot(fl, b) @ perp, chi))
            Ww.append(wi / nbeta)
    WW = np.einsum('n,nai,nbj->aibj', np.array(Ww), np.array(Wm), np.array(Wm))
    Z = np.array([_rot(n0, a) for a in al])
    ZZ = np.einsum('n,nai,nbj->aibj', wa, Z, Z)
    M = np.einsum('apbr,pq,rs,qisj->aibj', WW, R0, R0, ZZ)[:2, :, :2, :]
    D, S = _dipoles(Q, Hhf, eta, tet, phi)
    P = np.array([0.75 * np.einsum('aibj,i,j->ab', M, D[k], D[k].conj()) for k in range(8)])
    return P, S


# --------------------------------------------------------------------------
# (1) the convention: code vs first principles
# --------------------------------------------------------------------------
def test_single_crystal_blocks_match_first_principles():
    """The single-crystal blocks (Ham_mosaic at (1, 1, 1)) must equal the textbook
    blocks -- NOT their transpose.

    Before the 2026-08-26 convention fix they equalled the transpose, which is the
    same spectrum for one homogeneous non-dispersive layer but the opposite
    Faraday polarity everywhere else.
    """
    R = _lab_axes(**_GEO)
    Pref, Sref = _textbook_blocks(0.37, 28.4, 0.43, 37.0, 64.0, R)
    P, S = m5.Ham_mosaic(0.37, 28.4, 0.43, 64.0, 37.0, _GEO['tetr'], _GEO['phir'], _GEO['alfak'],
                         1.0, 1.0, 1.0, False)
    P, S = np.asarray(P), np.asarray(S)
    assert np.max(np.abs(S - Sref)) < 1e-12
    assert np.max(np.abs(P - Pref)) < 1e-12
    # the old (transposed) convention is genuinely different, i.e. this bites
    assert np.max(np.abs(P - np.transpose(Pref, (0, 2, 1)))) > 1e-3


def test_cms_blocks_match_first_principles():
    R = _lab_axes(_GEO['tetr'], _GEO['phir'], 0.0, cms=True)
    Pref, _ = _textbook_blocks(0.37, 28.4, 0.43, 37.0, 64.0, R)
    P, _ = m5.Ham_mosaic(0.37, 28.4, 0.43, 64.0, 37.0, _GEO['tetr'], _GEO['phir'], _GEO['alfak'],
                         1.0, 1.0, 1.0, True)
    assert np.max(np.abs(np.asarray(P) - Pref)) < 1e-12


def test_thin_intensities_are_convention_independent():
    """The scalar SMS intensity is |amplitude|^2, so it is the SAME before and
    after the convention fix -- it equals P[0,0] of the matrix form (checked on
    Ham_mono_classic, the eigensolver version kept for reuse)."""
    R = _lab_axes(**_GEO)
    Pref, _ = _textbook_blocks(0.37, 28.4, 0.43, 37.0, 64.0, R)
    I, _ = m5.Ham_mono_classic(0.37, 28.4, 0.43, 64.0, 37.0, _GEO['phir'], _GEO['tetr'])
    assert np.allclose(np.asarray(I), Pref[:, 0, 0].real, rtol=0, atol=1e-12)


def _summed_per_position(S, P, tol=1e-9):
    """Blocks summed over coincident lines: inside a degenerate level the
    eigenvectors are arbitrary, so only that sum is physical."""
    o = np.argsort(np.asarray(S, float))
    pos, blk = [], []
    for s, b in zip(np.asarray(S, float)[o], np.asarray(P)[o]):
        if pos and abs(s - pos[-1]) <= tol:
            blk[-1] = blk[-1] + b
        else:
            pos.append(s)
            blk.append(b)
    return np.array(pos), np.array(blk)


@pytest.mark.parametrize("cms", [False, True])
@pytest.mark.parametrize("eta", [0.35, 0.7, 0.99])
def test_zero_field_blocks_match_first_principles(eta, cms):
    """At H = 0 every level is a Kramers doublet: only the blocks summed over the
    coincident lines are physical, and by time reversal they carry no axial
    (Faraday) part. The general eig returned NON-orthogonal eigenvectors for such
    a doublet at these eta (not at all eta -- it depends on LAPACK's rounding),
    which left a spurious Faraday term of up to ~3% of the depth of a thick
    single-crystal spectrum; the Hermitian eigh cannot do that."""
    R = _lab_axes(_GEO['tetr'], _GEO['phir'], 0.0 if cms else _GEO['alfak'], cms=cms)
    P, S = m5.Ham_mosaic(0.8, 0.0, eta, 30.0, 40.0, _GEO['tetr'], _GEO['phir'],
                         _GEO['alfak'], 1.0, 1.0, 1.0, cms)
    Pref, Sref = _textbook_blocks(0.8, 0.0, eta, 40.0, 30.0, R)
    S, P = _summed_per_position(S, P)
    Sref, Pref = _summed_per_position(Sref, Pref)
    assert len(S) == len(Sref) == 2, "zero field: the lines sit at +-epsilon only"
    assert np.max(np.abs(S - Sref)) < 1e-12
    assert np.max(np.abs(P.imag)) < 1e-12, "a summed zero-field block has no Faraday part"
    assert np.max(np.abs(P - Pref)) < 1e-12


def test_pure_zeeman_blocks_equal_the_sextet_building_blocks():
    """At Q = 0 every transition block must equal the Sextet's own matrix for the
    same line, INCLUDING the sign of the Faraday term: sigma+ (Delta m = +1,
    lines 3 and 6) carries +i m_z J, sigma- carries -i m_z J."""
    R = _lab_axes(**_GEO)
    P, S = m5.Ham_mosaic(0.0, 33.0, 0.0, _PHI, _TET, _GEO['tetr'], _GEO['phir'],
                         _GEO['alfak'], 1.0, 1.0, 1.0, False)
    P, S = np.asarray(P), np.asarray(S)
    mx, my, mz = R @ _sph(_TET, _PHI)                # B_hf in lab coordinates
    Msig = m5._mhat_dm1_sym(mx, my)
    Mfar = 1.5 * 1.0 * 1j * mz * m5._J2              # S1 = 1 (A = Am = 1)
    Mpi = m5._mhat_dm0(mx, my)
    w1, w2, w3 = 0.25, 1 / 6, 1 / 12                 # I1, I2, I3 at I13 = 3, I = 1
    expect = [w1 * (Msig - Mfar), w2 * Mpi, w3 * (Msig + Mfar),
              w3 * (Msig - Mfar), w2 * Mpi, w1 * (Msig + Mfar)]
    allowed = [k for k in np.argsort(S) if abs(P[k, 0, 0] + P[k, 1, 1]) > 1e-9]
    assert len(allowed) == 6, "a pure Zeeman multiplet has 6 allowed lines"
    for n, k in enumerate(allowed):
        assert np.max(np.abs(P[k] - expect[n])) < 1e-12, f"line {n + 1} block mismatch"


# --------------------------------------------------------------------------
# (2) the exact limits, as full spectra through TImod
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mett,pol", [(0, 1.0), (0, 0.98), (0, 0.4), (1, 1.0)])
def test_single_crystal_limit_reproduces_hamilton_mc(mett, pol):
    a = _run(['Hamiltonian'], _HAM12 + [1.0, 1.0, 1.0], mett, pol)
    b = _run(['Hamilton_mc'], _HAM12, mett, pol)
    assert 1 - b.min() > 0.1, "the reference case must be genuinely thick"
    assert np.allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mett,pol", [(0, 1.0), (0, 0.98), (1, 1.0)])
def test_powder_limit_reproduces_hamilton_pc(mett, pol):
    a = _run(['Hamiltonian'], _HAM12 + [0.0, 0.0, 0.0], mett, pol)
    b = _run(['Hamilton_pc'], _HAM12[:9], mett, pol)
    assert 1 - b.min() > 0.1
    assert np.allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("stack", [['Hamiltonian', 'Sextet'],
                                   ['Hamiltonian', 'Layer', 'Sextet'],
                                   ['Sextet', 'Layer', 'Hamiltonian', 'Layer', 'Sextet']])
def test_isotropic_shortcut_is_exact_inside_a_stack(stack):
    """A fully disordered mosaic is a multiple of the identity, so the component
    takes the scalar (Beer--Lambert) path. That shortcut must be EXACT, also when
    the component shares a layer with a matrix component or sits between layers:
    a scalar factors out of the matrix exponential and of the layer product, and
    its dispersion is a global phase the Gram readout cancels."""
    sextet = _sextet_params(50.0, 30.0)

    def _params(names, ham):
        out = []
        for s in names:
            out += ham if s in ('Hamiltonian', 'Hamilton_pc') else ([] if s == 'Layer' else sextet)
        return out

    got = _run(stack, _params(stack, _HAM12 + [0.0, 0.0, 0.0]))
    ref_stack = [('Hamilton_pc' if s == 'Hamiltonian' else s) for s in stack]
    ref = _run(ref_stack, _params(ref_stack, _HAM12[:9]))
    assert np.allclose(got, ref, rtol=0, atol=1e-13)


def test_disordered_azimuth_and_axis_is_isotropic_whatever_am_is():
    """At A = Ah = 0 the wobble is isotropic and the azimuth uniform, so even a
    polar (Am != 0) mosaic averages to the random powder: the Faraday term needs
    a net axis along the beam, which a fiber texture about h cannot provide."""
    ref = _run(['Hamilton_pc'], _HAM12[:9])
    for Am in (0.0, 0.7, -1.0):
        assert np.allclose(_run(['Hamiltonian'], _HAM12 + [0.0, Am, 0.0]), ref,
                           rtol=0, atol=1e-13)


def test_powder_limit_ignores_the_reference_orientation():
    """At (0, 0, 0) the crystal orientation is uniformly random, so the reference
    angles theta/phi/alpha_k must have NO effect at all."""
    ref = _run(['Hamiltonian'], _HAM12 + [0.0, 0.0, 0.0])
    moved = _run(['Hamiltonian'], _HAM12[:9] + [77.0, -13.0, 155.0] + [0.0, 0.0, 0.0])
    assert np.array_equal(ref, moved)


def _sextet_angles(cms=False):
    """Sextet (theta_k, phi_h) equivalent to the pure-Zeeman Hamiltonian geometry."""
    R = _lab_axes(_GEO['tetr'], _GEO['phir'], 0.0 if cms else _GEO['alfak'], cms=cms)
    m = R @ _sph(_TET, _PHI)
    return np.degrees(np.arccos(np.clip(m[2], -1, 1))), np.degrees(np.arctan2(m[1], m[0])), m


def _sextet_params(th_k, ph_h, A=1.0, Am=1.0, T=8.0, delta=0.0, H=33.0):
    # T, delta, eps, H, L, G, theta_k, phi_h, A, Am, a+, a-, GH, I1/I3
    return [T, delta, 0.0, H, 0.098, 0.12, th_k, ph_h, A, Am, 0.0, 0.0, 0.0, 3.0]


@pytest.mark.parametrize("pol", [1.0, 0.98])
def test_zeeman_single_crystal_spectrum_equals_magnetised_sextet(pol):
    th_k, ph_h, _ = _sextet_angles()
    a = _run(['Hamiltonian'], _HAMQ0 + [1.0, 1.0, 1.0], 0, pol)
    b = _run(['Sextet'], _sextet_params(th_k, ph_h), 0, pol)
    flipped = _run(['Sextet'], _sextet_params(th_k, ph_h, Am=-1.0), 0, pol)
    assert np.allclose(a, b, rtol=0, atol=1e-11)
    # the Faraday sign is a real observable here: the flipped sextet differs
    assert np.max(np.abs(b - flipped)) > 1e-3


def test_zeeman_single_crystal_spectrum_equals_sextet_under_cms():
    th_k, ph_h, _ = _sextet_angles(cms=True)
    a = _run(['Hamiltonian'], _HAMQ0 + [1.0, 1.0, 1.0], 1, 1.0)
    b = _run(['Sextet'], _sextet_params(th_k, ph_h), 1, 1.0)
    assert np.allclose(a, b, rtol=0, atol=1e-11)


@pytest.mark.parametrize("Atex", [1.0, 0.5, -0.3])
def test_fiber_texture_limit_equals_the_textured_sextet(Atex):
    """Q = 0 and Ah = 0: a fiber texture about h, i.e. exactly the textured
    Sextet with its axis along h and A_eff = A * P2(cos theta_BH)."""
    _, _, m = _sextet_angles()
    A_eff = Atex * 0.5 * (3 * m[0] ** 2 - 1)          # m[0] = cos(angle to h)
    a = _run(['Hamiltonian'], _HAMQ0 + [Atex, 0.0, 0.0])
    b = _run(['Sextet'], _sextet_params(90.0, 0.0, A=A_eff, Am=0.0))   # axis || h
    assert np.allclose(a, b, rtol=0, atol=1e-11)


# --------------------------------------------------------------------------
# (3) the layer test: the relative Faraday polarity against another component
# --------------------------------------------------------------------------
_SEX2 = [6.0, 0.2, 0.0, 30.0, 0.098, 0.15, 65.0, 25.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0]


@pytest.mark.parametrize("stack", [['Hamiltonian', 'Layer', 'Sextet'],
                                   ['Hamiltonian', 'Sextet']])
def test_hamiltonian_layer_equals_the_sextet_it_reduces_to(stack):
    """A Hamiltonian layer in its sextet limit, stacked on (or mixed with) a
    magnetised Sextet layer, must give exactly what TWO such Sextets give.

    This is the geometry the old conjugated convention got wrong: the two
    components then carried opposite Faraday polarity, which is a first-order
    effect here (the deviation below is a large fraction of the line depth),
    unlike in a single homogeneous layer where it nearly cancels.
    """
    th_k, ph_h, _ = _sextet_angles()
    sex = _sextet_params(th_k, ph_h)
    sex_flipped = _sextet_params(th_k, ph_h, Am=-1.0)
    ref_stack = [('Sextet' if s == 'Hamiltonian' else s) for s in stack]
    mixed = _run(stack, _HAMQ0 + [1.0, 1.0, 1.0] + _SEX2)
    two_sextets = _run(ref_stack, sex + _SEX2)
    old_convention = _run(ref_stack, sex_flipped + _SEX2)
    assert np.allclose(mixed, two_sextets, rtol=0, atol=1e-11)
    # ... and the wrong polarity would have been a big, visible error
    assert np.max(np.abs(old_convention - two_sextets)) > 0.1 * (1 - two_sextets.min())


def test_layer_order_still_matters():
    """Sanity for the test above: the stack is genuinely non-commuting, so the
    agreement is not the trivial consequence of an order-blind readout."""
    a = _run(['Hamiltonian', 'Layer', 'Sextet'], _HAMQ0 + [1.0, 1.0, 1.0] + _SEX2)
    b = _run(['Sextet', 'Layer', 'Hamiltonian'], _SEX2 + _HAMQ0 + [1.0, 1.0, 1.0])
    assert np.max(np.abs(a - b)) > 1e-3


# --------------------------------------------------------------------------
# (4) the closed-form orientation average vs brute-force integration
# --------------------------------------------------------------------------
_ODF_CASES = [(0.7, 0.5, 0.6), (0.3, -0.4, 0.2), (-0.4, 0.1, 0.9),
              (0.0, 0.0, 0.5), (1.0, 0.0, 0.35), (0.55, 0.0, 0.0)]


@pytest.mark.parametrize("Atex,Am,Ah", _ODF_CASES)
@pytest.mark.parametrize("generic", [False, True])
def test_closed_form_average_matches_brute_force(Atex, Am, Ah, generic):
    R0 = _lab_axes(**_GEO)
    P, _ = m5.Ham_mosaic(0.37, 28.4, 0.43, 64.0, 37.0, _GEO['tetr'], _GEO['phir'],
                         _GEO['alfak'], Atex, Am, Ah, False)
    Pb, _ = _brute_force_blocks(0.37, 28.4, 0.43, 37.0, 64.0, R0,
                                np.array([1.0, 0.0, 0.0]), Atex, Am, Ah, generic=generic)
    assert np.max(np.abs(np.asarray(P) - Pb)) < 1e-12


@pytest.mark.parametrize("Atex,Am,Ah", [(0.7, 0.5, 0.6), (0.2, -0.3, 0.4)])
def test_closed_form_average_matches_brute_force_cms(Atex, Am, Ah):
    """Under CMS the mosaic is textured about the BEAM (the axis (tetr, phir)
    points at), not about a transverse direction -- an unpolarized source has
    none."""
    R0 = _lab_axes(_GEO['tetr'], _GEO['phir'], 0.0, cms=True)
    P, _ = m5.Ham_mosaic(0.37, 28.4, 0.43, 64.0, 37.0, _GEO['tetr'], _GEO['phir'],
                         _GEO['alfak'], Atex, Am, Ah, True)
    Pb, _ = _brute_force_blocks(0.37, 28.4, 0.43, 37.0, 64.0, R0,
                                np.array([0.0, 0.0, 1.0]), Atex, Am, Ah)
    assert np.max(np.abs(np.asarray(P) - Pb)) < 1e-12


def test_powder_blocks_are_isotropic_and_match_ham_poly():
    P, _ = m5.Ham_mosaic(0.37, 28.4, 0.43, 64.0, 37.0, _GEO['tetr'], _GEO['phir'],
                         _GEO['alfak'], 0.0, 0.0, 0.0, False)
    I, _ = m5.Ham_poly_classic(0.37, 28.4, 0.43, 64.0, 37.0)
    P, I = np.asarray(P), np.asarray(I)
    assert np.allclose(I.sum(), 1.0, rtol=0, atol=1e-12)     # normalisation sum rule
    for k in range(8):
        assert np.max(np.abs(P[k] - I[k] * np.eye(2))) < 1e-14


# --------------------------------------------------------------------------
# (5) geometry invariants and parameter bookkeeping
# --------------------------------------------------------------------------
@pytest.mark.parametrize("order", [(1.0, 1.0, 1.0), (0.6, 0.3, 0.4)])
def test_alfak_is_redundant_under_cms_but_observable_under_sms(order):
    ref_cms = _run(['Hamiltonian'], _HAMQ0 + list(order), 1, 1.0)
    ref_sms = _run(['Hamiltonian'], _HAMQ0 + list(order), 0, 1.0)
    moved = False
    for alfak in (0.0, 90.0, 123.0, -60.0):
        par = list(_HAMQ0)
        par[11] = alfak
        assert np.array_equal(_run(['Hamiltonian'], par + list(order), 1, 1.0), ref_cms)
        moved |= not np.allclose(_run(['Hamiltonian'], par + list(order), 0, 1.0), ref_sms)
    assert moved, "under SMS alpha_k rotates the polarization basis and IS observable"


def test_alfak_drops_out_when_the_azimuth_is_disordered():
    """At Ah = 0 the crystal is uniformly spun about the reference axis, so the
    reference azimuth alpha_k can no longer matter, even under SMS -- to
    round-off: alpha_k still rotates the reference axes inside the average and
    drops out of it only analytically (it moves the spectrum by ~5e-2 at Ah = 1)."""
    ref = _run(['Hamiltonian'], _HAMQ0 + [0.5, 0.2, 0.0], 0, 1.0)
    par = list(_HAMQ0)
    par[11] = 137.0
    assert np.allclose(_run(['Hamiltonian'], par + [0.5, 0.2, 0.0], 0, 1.0), ref,
                       rtol=0, atol=1e-13)


def test_parameter_block_is_fifteen_slots_wide():
    """A following component must read its own parameters: a scalar Singlet after
    a Hamiltonian factorises exactly into the two separate spectra."""
    singlet = [4.0, 0.0, 0.098, 0.1]
    ham = _HAMQ0 + [0.6, 0.3, 0.4]
    together = _run(['Hamiltonian', 'Singlet'], ham + singlet)
    apart = _run(['Hamiltonian'], ham) * _run(['Singlet'], singlet)
    assert np.allclose(together, apart, rtol=0, atol=1e-12)


def test_distribution_bookkeeping():
    """A degenerate distribution (L == R) over a Hamiltonian parameter splits the
    component into identical copies and must reproduce it exactly -- which only
    works if the Distr machinery knows the 15-slot footprint."""
    ham = _HAMQ0 + [0.6, 0.3, 0.4]
    plain = _run(['Hamiltonian'], ham)
    m5.COMPLEX_VOIGT_METHOD = 'pseudo'
    distributed = np.asarray(
        m5.TImod(_E, np.array(ham + [3.0, 33.0, 33.0, 3.0, 0.0], float),
                 np.array(['Hamiltonian', 'Distr']), _E, 0.0, 1.0, np.array([]),
                 ['X'], [], Met=-1, Mett=0, sms_pol=0.98), dtype=float)
    assert np.allclose(distributed, plain, rtol=0, atol=1e-12)


def test_line_position_markers_match_the_single_crystal():
    """The order parameters redistribute intensity, never position, so pos_ac
    must return the same 8 lines as the model it replaces."""
    from syncmoss.models_positions import pos_ac
    base = [0.0] * 8                                  # baseline slots
    got = pos_ac(np.array(base + _HAM12 + [0.6, 0.3, 0.4], float),
                 ['Hamiltonian'], np.array([]), Met=-1)
    ref = pos_ac(np.array(base + _HAM12, float), ['Hamilton_mc'], np.array([]), Met=-1)
    assert np.allclose(np.asarray(got, float), np.asarray(ref, float), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# (6) the closed form (Szymanski 2026) against the eigensolver
# --------------------------------------------------------------------------
_B_TO_b = gex * mun * c / E0_J          # the excited Zeeman scale b per tesla, mm/s


def _tensors_ref(Q, Hhf, eta, phi, tet):
    """First principles: M = (3/4) D D^dagger per line, = It + i [Gc]x."""
    D, S = _dipoles(Q, Hhf, eta, tet, phi)
    return np.array([0.75 * np.outer(d, d.conj()) for d in D]), S


def _tensors(Q, Hhf, eta, phi, tet):
    """The same M from the closed-form core."""
    It, Gc, S = m5._ham_tensor_core(Q, Hhf, eta, phi, tet)
    It, Gc = np.asarray(It), np.asarray(Gc)
    E = np.zeros((8, 3, 3))
    E[:, 0, 1], E[:, 1, 0] = Gc[:, 2], -Gc[:, 2]
    E[:, 2, 0], E[:, 0, 2] = Gc[:, 1], -Gc[:, 1]
    E[:, 1, 2], E[:, 2, 1] = Gc[:, 0], -Gc[:, 0]
    return It + 1j * E, np.asarray(S)


def _spectrum_error(Q, Hhf, eta, phi, tet):
    """The 8 lines of M on natural-width Lorentzians -- every consumer (thin
    intensities, 2x2 blocks, the mosaic average) is linear in M. Close lines
    differ between methods only in how they share their intensity, which no
    spectrum sees, so a spectrum compares them without a merging rule.
    Returns (error / peak, worst position error in mm/s)."""
    M, S = _tensors(Q, Hhf, eta, phi, tet)
    assert np.all(np.isfinite(M)) and np.all(np.isfinite(S))
    Mr, Sr = _tensors_ref(Q, Hhf, eta, phi, tet)
    grid = np.linspace(Sr.min() - 1.0, Sr.max() + 1.0, 801)

    def spectrum(M_, S_):
        L = (NAT_WIDTH / 2 / np.pi) / ((grid[None, :] - S_[:, None]) ** 2 + (NAT_WIDTH / 2) ** 2)
        return np.einsum('kx,kab->xab', L, M_)

    ref = spectrum(Mr, Sr)
    return (np.max(np.abs(spectrum(M, S) - ref)) / np.max(np.abs(ref)),
            np.max(np.abs(np.sort(S) - np.sort(Sr))))


def test_closed_form_lines_match_the_eigensolver_one_by_one():
    """Generic geometries, line by line in the same order: the closed-form core
    against first principles, and Ham_mosaic against Ham_mosaic_classic (the
    eigensolver version), SMS and CMS, crystal and mosaic."""
    rng = np.random.default_rng(2026)
    for _ in range(100):
        Q, H, eta = rng.uniform(-3, 3), rng.uniform(-60, 60), rng.uniform(0, 1)
        phi, tet = rng.uniform(0, 360), rng.uniform(0, 180)
        tetr, phir, alfak = rng.uniform(0, 180), rng.uniform(0, 360), rng.uniform(0, 360)
        mosaic = (rng.uniform(-0.5, 1), rng.uniform(-1, 1), rng.uniform(0, 1))
        M, S = _tensors(Q, H, eta, phi, tet)
        Mr, Sr = _tensors_ref(Q, H, eta, phi, tet)
        assert np.max(np.abs(S - Sr)) < 1e-12
        assert np.max(np.abs(M - Mr)) < 1e-12
        for order in ((1.0, 1.0, 1.0), mosaic):
            for cms in (False, True):
                args = (Q, H, eta, phi, tet, tetr, phir, alfak) + order + (cms,)
                P, S = m5.Ham_mosaic(*args)
                Pc, Sc = m5.Ham_mosaic_classic(*args)
                assert np.max(np.abs(np.asarray(S) - np.asarray(Sc))) < 1e-12
                assert np.max(np.abs(np.asarray(P) - np.asarray(Pc))) < 1e-12


# the paper's curve (40): m along the second principal axis, field per mm/s of V_zz
_ETA40 = 0.5
_V40 = np.sqrt(72 / (13 * _ETA40 ** 2 + 30 * _ETA40 + 9))
_B40 = np.sqrt(0.4 - _V40 ** 2 * (3 + _ETA40 ** 2) / 60) / _V40 / abs(_B_TO_b)

# coincident and close excited levels: (Q, H, eta, phi, theta)
_CLOSE_LEVELS = {
    "B = 0, Kramers doublets": (0.8, 0.0, 0.35, 30.0, 40.0),
    "B = 0, axial EFG": (0.8, 0.0, 0.0, 0.0, 0.0),
    "Q = H = 0, one line": (0.0, 0.0, 0.0, 0.0, 0.0),
    "theta 90, eta 0, 1 T: the +-3/2 pair splits as B^3": (0.6, 1.0, 0.0, 0.0, 90.0),
    "the same, 0.01 T": (0.6, 0.01, 0.0, 0.0, 90.0),
    "the same, 0.001 T": (0.6, 0.001, 0.0, 0.0, 90.0),
    "the near-degenerate case of test_sms_theory": (0.31, 0.02, 0.0, 0.0, 90.0),
    "1e-6 T, generic angles": (0.6, 1e-6, 0.3, 64.0, 37.0),
    "theta 0, eta 0, at the crossing |b/V_zz| = 1/2": (0.6, 0.5 * 0.6 / abs(_B_TO_b), 0.0, 0.0, 0.0),
    "the same, off by 1e-6": (0.6, 0.5 * 0.6 / abs(_B_TO_b) * (1 + 1e-6), 0.0, 0.0, 0.0),
    "theta 0, eta 0, at the crossing |b/V_zz| = 1/4 (the middle pair)": (0.6, 0.25 * 0.6 / abs(_B_TO_b), 0.0, 0.0, 0.0),
    "the paper's point s0: eta 1, theta 45, phi 90, b/V_zz = 1/3": (0.6, 0.6 / 3 / _B_TO_b, 1.0, 90.0, 45.0),
    "the paper's curve (40), eta 0.5": (1.0, _B40, _ETA40, 90.0, 90.0),
}


@pytest.mark.parametrize("case", list(_CLOSE_LEVELS))
def test_closed_form_is_exact_where_levels_coincide(case):
    """Where two excited levels coincide or nearly do, the paper's formulas give
    0/0: NaN, or intensities off by up to the whole line. With the pair formulas
    every spectrum follows first principles to round-off, and every position."""
    Q, H, eta, phi, tet = _CLOSE_LEVELS[case]
    _, Sr = _dipoles(Q, H, eta, tet, phi)
    assert np.min(np.diff(Sr[:4])) <= 0.05 * (Sr[3] - Sr[0]), "the case must have close levels"
    err, dpos = _spectrum_error(Q, H, eta, phi, tet)
    assert dpos < 1e-12
    assert err < 1e-12


def test_marker_positions_match_first_principles():
    """models_positions keeps its own copy of the closed form, only up to the
    levels: the 8 positions of first principles, line by line on generic sets.
    Where two levels coincide they carry the rounding of the quartic's double
    root, sqrt(eps) of the level scale -- the core removes it with the
    determinant; a marker cannot show it."""
    from syncmoss.models_positions import Ham_positions
    rng = np.random.default_rng(7)
    for _ in range(100):
        Q, H, eta = rng.uniform(-3, 3), rng.uniform(-60, 60), rng.uniform(0, 1)
        phi, tet = rng.uniform(0, 360), rng.uniform(0, 180)
        _, Sr = _dipoles(Q, H, eta, tet, phi)
        assert np.max(np.abs(np.asarray(Ham_positions(Q, H, eta, phi, tet)) - Sr)) < 1e-12
    for case, (Q, H, eta, phi, tet) in _CLOSE_LEVELS.items():
        _, Sr = _dipoles(Q, H, eta, tet, phi)
        S = np.asarray(Ham_positions(Q, H, eta, phi, tet))
        tol = 2 * np.sqrt(np.finfo(float).eps) * np.max(np.abs(Sr))
        assert np.max(np.abs(np.sort(S) - np.sort(Sr))) <= tol, case


@pytest.mark.parametrize("Q,eta,phi,tet,Bs", [
    (0.6, 0.0, 0.0, 90.0, np.linspace(-2e-3, 2e-3, 41)),
    (0.6, 0.0, 0.0, 0.0, 0.5 * 0.6 / abs(_B_TO_b) * (1 + np.linspace(-1e-9, 1e-9, 41))),
    (1.0, _ETA40, 90.0, 90.0, _B40 * (1 + np.linspace(-1e-6, 1e-6, 41))),
], ids=["through B = 0", "through a crossing", "through the curve (40)"])
def test_closed_form_is_smooth_through_coincidences(Q, eta, phi, tet, Bs):
    """Nothing in the closed form switches on a threshold: along a scan of B
    through a coincidence the spectrum follows first principles to round-off at
    every step -- what a finite-difference Jacobian needs."""
    for B in Bs:
        err, _ = _spectrum_error(Q, B, eta, phi, tet)
        assert err < 1e-12, f"B = {B!r}"


def test_closed_form_is_smooth_where_the_closest_pair_changes():
    """The one discrete choice, which pair of levels is the closest, changes
    where both choices are accurate: no step in the spectrum across it."""
    Q, eta, phi, tet = 0.6, 0.3, 64.0, 37.0
    Bs = np.linspace(1.36, 1.375, 61)
    closest = {int(np.argmin(np.diff(_dipoles(Q, B, eta, tet, phi)[1][:4]))) for B in Bs}
    assert len(closest) == 2, "the scan must cross a change of the closest pair"
    for B in Bs:
        err, _ = _spectrum_error(Q, B, eta, phi, tet)
        assert err < 1e-12, f"B = {B!r}"


@pytest.mark.parametrize("params,mett", [
    (_HAM12 + [1.0, 1.0, 1.0], 0), (_HAM12 + [1.0, 1.0, 1.0], 1),
    (_HAM12 + [0.6, 0.3, 0.4], 0), (_HAM12 + [0.6, 0.3, 0.4], 1),
    (_HAM12 + [0.0, 0.0, 0.0], 0),
    ([8.0, 0.0, 0.8, 0.0, 0.098, 0.12, 0.35, 40.0, 30.0, 54.0, 71.0, 23.0, 1.0, 1.0, 1.0], 0),
    ([8.0, 0.0, 0.6, 0.5 * 0.6 / abs(_B_TO_b), 0.098, 0.12, 0.0, 0.0, 0.0, 54.0, 71.0, 23.0,
      1.0, 1.0, 1.0], 0),
], ids=["crystal SMS", "crystal CMS", "mosaic SMS", "mosaic CMS", "powder SMS", "B = 0 SMS",
        "at a crossing SMS"])
def test_closed_form_spectra_equal_the_eigensolver_ones(params, mett, monkeypatch):
    """Thick 'Hamiltonian' spectra through TImod are those of the eigensolver
    version Ham_mosaic_classic to round-off."""
    new = _run(['Hamiltonian'], params, mett)
    monkeypatch.setattr(m5, 'Ham_mosaic', m5.Ham_mosaic_classic)
    old = _run(['Hamiltonian'], params, mett)
    assert 1 - old.min() > 0.1, "the reference case must be genuinely thick"
    assert np.allclose(new, old, rtol=0, atol=1e-12)
