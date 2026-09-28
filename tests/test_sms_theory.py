"""Tests for the theoretical (simulated 57FeBO3) SMS instrumental function.

Three groups:

* the physics -- the hyperfine Hamiltonian of ``sms_theory`` against the one
  ``models.Ham_mono`` ('Hamiltonian' component) uses, which is the validated
  implementation; plus the module's own selftest suite;
* the INS encoding -- round-trips, the legacy form still being recognised as
  legacy, and the shape/tail/centroid helpers agreeing with quadrature;
* the integration -- models.TImod, models.limits, models_positions.pos_ac and
  the .dat metadata all handling the new forms while leaving the legacy path
  bit-identical.

Pure NumPy/Numba path -- no Qt, no multiprocessing pool.
"""
import os

import numpy as np
import pytest

import syncmoss.models as m5
import syncmoss.models_positions as mp5
import syncmoss.sms_theory as st
from syncmoss.constants import (MMPS_TO_NEV, NAT_WIDTH, ALPHA_FE_FIELD,
                                LINE_SHIFT_16, LINE_SHIFT_34,
                                number_of_baseline_parameters as NB)


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


_LEGACY_INS = np.array([
    0.44929437564516517, -0.04402597813422836, 0.4491092302535263,
    -0.1873563911726706, 0.06604033956754407, 0.7568280670234774,
    -0.16807996586604437, -0.10974723859984138, 0.4748812233249404])
_LEGACY_X0, _LEGACY_MULCO = 0.00433441914897599, 3.1552701198151407


# ======================================================================
#  the physics: same Hamiltonian as the 'Hamiltonian' model
# ======================================================================

def _sms_lines(Q_mmps, Hhf, tet_deg, phi_deg, tetr_deg, phir_deg, eta=0.0):
    """Line positions [mm/s] and intensities from sms_theory, in Ham_mono terms.

    models.Ham_mono  H_e = (Q/12)[3Iz^2 - I(I+1) + eta(Ix^2-Iy^2)] - gex mun H (m.I)
    sms_theory       H_e = (dEQ/6)[3Iz^2 - I(I+1)]                 - gex MU_N B (m.I)
    so dEQ = Q/2: sms_theory's dEQ is the quadrupole SPLITTING (the |+-3/2> to
    |+-1/2> separation), which is what the doublet line spacing measures.

    Intensities: Ham_mono returns pi*|<e|I.h|g>|^2, sms_theory carries the factor
    3/(2(2Ig+1)) = 3/4 on the amplitude, so I[k] == (3/4)|W(b)|^2.
    """
    t, p = np.deg2rad(tet_deg), np.deg2rad(phi_deg)
    m_hat = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    site = st.HyperfineSite(Hhf, m_hat, (Q_mmps / 2.0) * MMPS_TO_NEV, 0.0, eta)

    tr, pr = np.deg2rad(tetr_deg), np.deg2rad(phir_deg)
    b = np.array([np.sin(tr) * np.cos(pr), np.sin(tr) * np.sin(pr), np.cos(tr)],
                 dtype=complex)
    W = np.einsum('i,iba->ba', b, site.t)
    return (site.dE / MMPS_TO_NEV).ravel(), (0.75 * np.abs(W) ** 2).ravel()


def _merge_degenerate(S, I, tol=1e-9):
    """Sort by position and merge coincident lines.

    Inside a degenerate multiplet the eigenvectors are arbitrary, so only the
    summed intensity of a group is physical -- and that is all the transmission
    integral can see, since the lines coincide.
    """
    o = np.argsort(np.asarray(S, float))
    S, I = np.asarray(S, float)[o], np.asarray(I, float)[o]
    pos, wgt = [], []
    for s, i in zip(S, I):
        if pos and abs(s - pos[-1]) <= tol:
            wgt[-1] += i
        else:
            pos.append(s)
            wgt.append(i)
    return np.array(pos), np.array(wgt)


@pytest.mark.parametrize("Q,H,tet,phi,tetr,phir", [
    (0.0, ALPHA_FE_FIELD, 0, 0, 90, 0),        # pure Zeeman, h perpendicular
    (0.0, ALPHA_FE_FIELD, 0, 0, 0, 0),         # pure Zeeman, h along the field
    (0.5, 0.0, 0, 0, 55, 33),                  # pure quadrupole
    (-0.4, 0.5, 90, 0, 90, 90),                # combined at 90 deg: the FeBO3 case
    (-0.2, 1.0, 90, 0, 90, 0),
    (0.37, 2.3, 41, 27, 63, 111),              # generic angles
    (-0.9, 17.0, 71, 13, 29, 47),
    (0.31, 0.02, 90, 0, 90, 90),               # near-degenerate small field
])
def test_hyperfine_site_matches_hamiltonian_model(Q, H, tet, phi, tetr, phir):
    """sms_theory's site == models.Ham_mono, to machine precision."""
    Im, Sm = m5.Ham_mono(Q, H, 0.0, phi, tet, phir, tetr)
    Ss, Is = _sms_lines(Q, H, tet, phi, tetr, phir)
    Sm, Im = _merge_degenerate(Sm, Im)
    Ss, Is = _merge_degenerate(Ss, Is)

    assert Sm.shape == Ss.shape
    assert np.max(np.abs(Sm - Ss)) < 1e-12, "line positions differ"
    assert np.max(np.abs(Im - Is)) < 1e-12 * max(np.max(Im), 1e-30), \
        "line intensities differ"
    # unit total oscillator strength, for any geometry
    assert abs(Im.sum() - 1.0) < 1e-12
    assert abs(Is.sum() - 1.0) < 1e-12


def test_sextet_reproduces_syncmoss_line_positions():
    """At the alpha-Fe field the site must give SYNCmoss's own sextet.

    The radiation field is taken off both axes (tetr = 55 deg) so that all three
    q channels contribute: with b perpendicular to B only the four Delta m = +-1
    lines survive, which is the very selection rule the SMS filter theorem
    exploits but not what this test is about.
    """
    S, I = _sms_lines(0.0, ALPHA_FE_FIELD, 0, 0, 55, 0)
    keep = I > 1e-12
    pos = np.sort(S[keep])
    assert keep.sum() == 6
    assert abs(pos[-1] - LINE_SHIFT_16 * ALPHA_FE_FIELD) < 1e-12
    assert abs(pos[3] - LINE_SHIFT_34 * ALPHA_FE_FIELD) < 1e-12


@pytest.mark.parametrize("Q,H,tet,phi,tetr,phir,eta", [
    (0.5, 0.0, 0, 0, 55, 33, 0.35),            # pure quadrupole, asymmetric EFG
    (-0.4, 0.5, 90, 0, 90, 90, 0.20),          # the FeBO3 geometry, eta != 0
    (0.37, 2.3, 41, 27, 63, 111, 0.60),        # generic angles
    (-0.9, 17.0, 71, 13, 29, 47, 1.00),        # eta at its maximum
])
def test_efg_asymmetry_matches_hamiltonian_model(Q, H, tet, phi, tetr, phir, eta):
    """The EFG asymmetry term must agree with models.Ham_mono's too.

    eta = (V_xx - V_yy)/V_zz is a material constant of the crystal, not a
    per-spectrum knob, and it is zero for an ideal FeBO3 site (symmetry -3). It
    is implemented so that assumption can be TESTED, which means the term itself
    has to be right.
    """
    Im, Sm = m5.Ham_mono(Q, H, eta, phi, tet, phir, tetr)
    Ss, Is = _sms_lines(Q, H, tet, phi, tetr, phir, eta=eta)
    Sm, Im = _merge_degenerate(Sm, Im)
    Ss, Is = _merge_degenerate(Ss, Is)
    assert Sm.shape == Ss.shape
    assert np.max(np.abs(Sm - Ss)) < 1e-12, "line positions differ"
    assert np.max(np.abs(Im - Is)) < 1e-12 * max(np.max(Im), 1e-30)
    assert Im.sum() == pytest.approx(1.0, abs=1e-12)


def test_eta_changes_the_zero_field_splitting_as_it_should():
    """At zero field the two levels separate by dEQ*sqrt(1+eta^2/3): that is the
    relation that makes dEQ the COUPLING constant and not the splitting."""
    for eta in (0.0, 0.3, 0.7, 1.0):
        site = st.HyperfineSite(0.0, np.array([0.0, 0.0, 1.0]),
                                0.4 * MMPS_TO_NEV, 0.0, eta)
        lines = np.unique(np.round(site.dE.ravel() / MMPS_TO_NEV, 10))
        assert len(lines) == 2
        assert np.ptp(lines) == pytest.approx(0.4 * np.sqrt(1 + eta ** 2 / 3),
                                              rel=1e-9)


def test_quadrupole_convention_is_the_doublet_splitting():
    """dEQ is the |+-3/2> - |+-1/2> separation, i.e. the doublet line spacing."""
    for dEQ in (0.2, -0.4):
        site = st.HyperfineSite(0.0, np.array([0.0, 0.0, 1.0]), dEQ * MMPS_TO_NEV)
        lines = np.unique(np.round(site.dE.ravel() / MMPS_TO_NEV, 12))
        assert len(lines) == 2
        assert abs(np.ptp(lines) - abs(dEQ)) < 1e-12


@pytest.mark.slow
def test_sms_theory_selftest():
    """The module's full validation suite (filter theorem, dynamical solver,
    sum rules, tail exponents, rational reduction, absorber-area invariance)."""
    assert st.selftest(verbose=False)


# ======================================================================
#  B_s(T), calibrated on the ESRF (temperature, angle) study
# ======================================================================

# The nine temperatures of that study and the B_s the measured COUNT RATES give
# (Yaroslavtsev & Chumakov 2022 raw data; see docs/sms_instrumental_function.md).
_BS_OF_T = [
    (75.675, 2.393), (75.725, 1.547), (75.775, 1.089), (75.825, 0.812),
    (75.875, 0.631), (75.925, 0.505), (76.025, 0.347), (76.120, 0.258),
    (76.230, 0.193),
]


@pytest.mark.parametrize("T,B", _BS_OF_T)
def test_bs_from_temperature_reproduces_the_calibration(T, B):
    assert st.Bs_from_temperature(T) == pytest.approx(B, abs=0.002)


def test_bs_from_temperature_is_monotone_and_clamped():
    T = np.linspace(75.0, 77.0, 401)
    B = st.Bs_from_temperature(T)
    assert np.all(np.diff(B) <= 1e-12), "B_s must fall as the crystal warms"
    lo, hi = st.FEBO3_BS_CALIBRATION['T_lo'], st.FEBO3_BS_CALIBRATION['T_hi']
    # outside the measured range the law is held, never extrapolated: a critical
    # power law diverges at T_N and means nothing above the last measured point
    assert st.Bs_from_temperature(lo - 5.0) == pytest.approx(st.Bs_from_temperature(lo))
    assert st.Bs_from_temperature(hi + 5.0) == pytest.approx(st.Bs_from_temperature(hi))
    assert np.all(B > 0)


@pytest.mark.parametrize("T,_B", _BS_OF_T)
def test_temperature_from_bs_round_trips(T, _B):
    assert st.temperature_from_Bs(st.Bs_from_temperature(T)) == pytest.approx(T, abs=1e-6)


def test_bs_law_covers_the_operating_range():
    """The published series spans more than a decade in B_s; a law that does not
    is not describing this source."""
    lo, hi = st.FEBO3_BS_CALIBRATION['T_lo'], st.FEBO3_BS_CALIBRATION['T_hi']
    assert st.Bs_from_temperature(lo) / st.Bs_from_temperature(hi) > 10.0


# ======================================================================
#  the INS encoding
# ======================================================================

def test_legacy_ins_is_recognised_as_legacy():
    assert st.ins_kind(_LEGACY_INS) == st.KIND_GAUSS
    assert st.ins_kind([]) == st.KIND_GAUSS
    assert st.ins_kind([0.3]) == st.KIND_GAUSS
    assert st.ins_kind([0.3, 0.0, 1.0]) == st.KIND_GAUSS


def test_physical_ins_round_trip():
    ins = st.encode_physical(theta_urad=63.5, B_s=0.42, dEQ=-0.17, shift=0.011)
    assert st.ins_kind(ins) == st.KIND_PHYSICAL
    got = st.decode_physical(ins)
    for k, v in (('theta_urad', 63.5), ('B_s', 0.42), ('dEQ', -0.17),
                 ('shift', 0.011)):
        assert got[k] == pytest.approx(v, abs=1e-15)
    # the untouched fields keep their documented defaults
    assert got['thickness_um'] == st.DEFAULT_THICKNESS_UM
    assert got['N_order'] == 1
    with pytest.raises(TypeError):
        st.encode_physical(not_a_field=1.0)


def test_rational_ins_round_trip():
    p = [0.0, 0.05, 0.02, 0.09, 0.8, 0.1, 0.2, 0.1, 0.3, 0.6]
    ins = st.encode_rational(p)
    assert st.ins_kind(ins) == st.KIND_RATIONAL
    assert np.allclose(st.decode_rational(ins), p)
    with pytest.raises(ValueError):
        st.encode_rational([1.0, 2.0])


@pytest.mark.parametrize("make", [
    lambda: _LEGACY_INS,
    lambda: st.encode_rational([0.0, 0.05, 0.03, 0.09, 1.0]),
    lambda: st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2),
])
def test_shape_is_a_unit_area_density(make):
    ins = make()
    v = np.linspace(-2.5, 2.5, 50001)
    S = st.ins_shape(ins, v)
    assert np.all(S >= 0)
    area = (np.trapezoid(S, v) + float(st.ins_tail_above(ins, v[-1]))
            + 1.0 - float(st.ins_tail_above(ins, v[0])))
    assert area == pytest.approx(1.0, abs=3e-3)


@pytest.mark.parametrize("make", [
    lambda: _LEGACY_INS,
    lambda: st.encode_rational([0.0, 0.05, 0.03, 0.09, 1.0]),
    lambda: st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2),
])
def test_tail_integral_is_the_primitive_of_the_shape(make):
    ins = make()
    v = np.linspace(-2.5, 2.5, 50001)
    S = st.ins_shape(ins, v)
    lo, hi = -0.31, 0.44
    band = (v >= lo) & (v <= hi)
    numeric = float(np.trapezoid(S[band], v[band]))
    closed = float(st.ins_tail_above(ins, lo)) - float(st.ins_tail_above(ins, hi))
    assert closed == pytest.approx(numeric, abs=2e-4)
    assert float(st.ins_tail_above(ins, -1e4)) == pytest.approx(1.0, abs=1e-5)
    assert float(st.ins_tail_above(ins, 1e4)) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("make", [
    lambda: _LEGACY_INS,
    lambda: st.encode_rational([0.05, 0.05, -0.02, 0.09, 1.0]),
    lambda: st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2),
])
def test_centroid_is_the_first_moment(make):
    ins = make()
    v = np.linspace(-6.0, 6.0, 240001)
    S = st.ins_shape(ins, v)
    numeric = float(np.trapezoid(v * S, v))
    assert float(st.ins_centroid(ins)) == pytest.approx(numeric, abs=2e-3)


def test_legacy_centroid_matches_the_formula_it_replaces():
    """ins_centroid must reproduce sum_i pos_i * amp_i^2 exactly."""
    expected = sum(_LEGACY_INS[3 * i + 1] * _LEGACY_INS[3 * i + 2] ** 2
                   for i in range(len(_LEGACY_INS) // 3))
    assert st.ins_centroid(_LEGACY_INS) == pytest.approx(expected, abs=1e-15)


def test_theoretical_tails_go_as_v_minus_4():
    """The sum rule sum_j c_j = 0 makes S ~ v^-4 -- the property a Gaussian sum
    cannot have, and the reason the theoretical shape is worth fitting."""
    ins = st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2)
    c = st._physical_table(ins)[0]
    for form in (ins, st.physical_to_rational(ins, n_terms=2)[0]):
        s1 = float(st.ins_shape(form, c + 20.0))
        s2 = float(st.ins_shape(form, c + 40.0))
        assert np.log(s1 / s2) / np.log(2.0) == pytest.approx(4.0, abs=0.05)


def test_gaussian_stand_in_follows_the_core_but_not_the_tails():
    ins = st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2)
    gauss, rms, _mx = st.physical_to_gaussians(ins, n_terms=3)
    assert st.ins_kind(gauss) == st.KIND_GAUSS
    assert np.sum(gauss[2::3] ** 2) == pytest.approx(1.0, abs=1e-12)
    assert rms < 0.05                                   # the core is reproduced
    c = st._physical_table(ins)[0]
    # ... and the wings are not: at 20 natural widths out the Gaussian sum has
    # collapsed by orders of magnitude relative to the real v^-4 tail
    assert (st.gaussian_sum(c + 20 * NAT_WIDTH, gauss)
            < 0.2 * float(st.ins_shape(ins, c + 20 * NAT_WIDTH)))


# ======================================================================
#  integration with the transmission integral
# ======================================================================

def _ti(ins, x0, MulCo, JN=40):
    pool = _SerialPool()
    x = np.linspace(-2.0, 2.0, 41)
    p = np.array([1.0e6, 0, 0, 0, 0, 0, 0, 0] + [5.0, 0.0, NAT_WIDTH, 0.0])
    return np.asarray(m5.TI(x, p, ['Singlet'], JN, pool, x0, MulCo, ins), float)


def test_legacy_transmission_integral_is_untouched():
    """The Gaussian branch of TImod must be bit-identical to what it was: the
    dispatch is a new `elif`, not a change to the old expression."""
    F = _ti(_LEGACY_INS, _LEGACY_X0, _LEGACY_MULCO)
    # recompute the legacy expression straight from the formula, node by node
    pool = _SerialPool()
    x = np.linspace(-2.0, 2.0, 41)
    p = np.array([1.0e6, 0, 0, 0, 0, 0, 0, 0] + [5.0, 0.0, NAT_WIDTH, 0.0])
    JN = 40
    E = np.linspace(-1 + 1e-3, 1 - 1e-3, JN)
    total = np.zeros_like(x)
    for EE in E:
        total += np.asarray(m5.TImod(x, p, ['Singlet'], EE, _LEGACY_X0,
                                     _LEGACY_MULCO, _LEGACY_INS, [0], [0]), float)
    expected = total * p[0] * (E[1] - E[0])
    assert np.allclose(F, expected, rtol=1e-12, atol=0)
    assert np.all(np.isfinite(F))


@pytest.mark.parametrize("make", [
    lambda: st.encode_rational([0.0, 0.05, 0.03, 0.09, 1.0]),
    lambda: st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2),
])
def test_theoretical_transmission_integral_runs_and_is_sane(make):
    ins = make()
    x0, MulCo = 0.0, 3.0
    F = _ti(ins, x0, MulCo)
    assert np.all(np.isfinite(F))
    assert F.max() > 0
    assert F.min() > 0                 # transmission, never negative
    assert F.min() < F.max()           # there IS an absorption dip


def _reference_transmission(ins, x, x0, MulCo, I_eff, delta, WL, WG, n_t=40001):
    r"""The transmission integral written out directly, as a reference.

        N(x) = p0 * int dt S(t) exp[-pi (G_nat/2) I Voigt(x + t - delta)]

    i.e. the source emits at velocity t with weight S(t) and therefore probes the
    absorber at x + t.  TI computes the same integral after substituting
    t = log((1+EE)/(1-EE))/MulCo + x0, so comparing the two checks the new branch's
    normalisation, its Jacobian and the SIGN of its velocity axis at once.
    ``t`` is integrated over exactly the range TI's truncated EE grid covers.
    """
    u_max = np.log((1 + (1 - 1e-3)) / (1 - (1 - 1e-3)))
    t = np.linspace(x0 - u_max / MulCo, x0 + u_max / MulCo, n_t)
    S = st.ins_shape(ins, t)
    out = np.empty(len(x))
    for i, xv in enumerate(x):
        voi = m5.Voight(WL * MulCo, WG * MulCo, (xv + t - delta) * MulCo)
        out[i] = np.trapezoid(
            S * np.exp(-np.pi * (NAT_WIDTH / 2 * MulCo) * I_eff * voi), t)
    return out


@pytest.mark.parametrize("make", [
    lambda: _LEGACY_INS,
    lambda: st.encode_rational([0.0, 0.05, 0.03, 0.09, 1.0]),
    lambda: st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2),
])
def test_transmission_integral_matches_the_convolution_it_stands_for(make):
    """TI must equal the explicit source-shape convolution, for every INS form.

    The legacy case is the control: it pins the reference implementation, so the
    two theoretical cases passing means the new TImod branch enters the integral
    with the same normalisation and the same velocity-axis sign as the Gaussian
    sum it replaces -- the two things that would otherwise fail silently, as a
    rescaled baseline and a mirrored source.
    """
    ins = make()
    x0, MulCo, JN = 0.0, 3.0, 600
    I_eff, delta, WL, WG = 5.0, 0.0, NAT_WIDTH, 0.0
    p0 = 1.0e6

    pool = _SerialPool()
    x = np.linspace(-1.5, 1.5, 21)
    p = np.array([p0, 0, 0, 0, 0, 0, 0, 0] + [I_eff, delta, WL, WG])
    got = np.asarray(m5.TI(x, p, ['Singlet'], JN, pool, x0, MulCo, ins), float)

    want = p0 * _reference_transmission(ins, x, x0, MulCo, I_eff, delta, WL, WG)
    # measured: 8e-13 (legacy -- the two quadratures agree to round-off) and
    # 6e-5 for both theoretical forms, where the v^-4 tails are resolved
    # differently by the log substitution than by the uniform reference grid.
    assert np.allclose(got, want, rtol=3e-4, atol=0), (
        f"max relative deviation {np.max(np.abs(got / want - 1)):.2e}")


def test_shifted_source_moves_the_dip_the_same_way_as_the_legacy_form():
    """A source line at +0.3 mm/s must displace the dip exactly as the legacy
    (width, position, amplitude) triple with the same position does."""
    shift = 0.3
    legacy = np.array([0.0, shift, 1.0])
    rat = st.encode_rational([shift, NAT_WIDTH / 2 * 0.02, shift,
                              NAT_WIDTH / 2 * 0.02, 1.0])
    Fg = _ti(legacy, 0.0, 3.0, JN=200)
    Fr = _ti(rat, 0.0, 3.0, JN=200)
    assert np.argmin(Fg) == np.argmin(Fr)


@pytest.mark.parametrize("make", [
    lambda: _LEGACY_INS,
    lambda: st.encode_rational([0.0, 0.05, 0.03, 0.09, 1.0]),
    lambda: st.encode_physical(theta_urad=70.0, B_s=0.5, dEQ=-0.2),
])
def test_limits_returns_a_usable_grid(make):
    x0, MulCo = m5.limits(_SerialPool(), 32, make())
    assert np.isfinite(x0) and np.isfinite(MulCo)
    assert abs(x0) <= 0.5
    assert 1.0 < MulCo < 6.0


def test_pos_ac_shifts_lines_by_the_instrumental_centroid():
    """models_positions.pos_ac subtracts the instrumental first moment; the new
    forms must go through the same path."""
    p = np.zeros(NB + 4)
    p[NB + 1] = 0.25                                  # a Singlet at +0.25 mm/s
    ins = st.encode_rational([0.12, 0.05, 0.12, 0.05, 1.0])
    pos_legacy = mp5.pos_ac(p, ['Singlet'], np.array([0.05, 0.12, 1.0]))
    pos_theory = mp5.pos_ac(p, ['Singlet'], ins)
    assert np.allclose(np.asarray(pos_legacy, float),
                       np.asarray(pos_theory, float), atol=1e-9)


# ======================================================================
#  .dat metadata: the #@INSacc marker
# ======================================================================

def _write_dat(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Converted by SYNCmoss\n")
        for line in lines:
            f.write(line + "\n")
        f.write("0.0\t100\n1.0\t100\n")
    return path


def test_dat_metadata_reads_the_accurate_marker(tmp_path):
    from syncmoss import instrumental_io as iio
    acc = st.encode_physical(theta_urad=55.0, B_s=0.44, dEQ=-0.13, shift=0.02)
    dat = _write_dat(str(tmp_path / "s.dat"), [
        "#@INSexp " + " ".join(str(float(v)) for v in _LEGACY_INS),
        f"#@INSint {_LEGACY_MULCO} {_LEGACY_X0}",
        "#@INSacc " + " ".join(str(float(v)) for v in acc),
    ])
    meta = iio.parse_dat_instrumental_metadata(dat)
    assert meta['has_insexp'] and meta['has_insint'] and meta['has_insacc']
    assert np.allclose(meta['INS'], _LEGACY_INS)
    assert np.allclose(meta['INSacc'], acc)
    # the conventional lines are still there for a reader that ignores #@INSacc
    assert len(iio.read_dat_metadata_lines(dat)) == 3


def test_dat_without_accurate_marker_is_unchanged(tmp_path):
    from syncmoss import instrumental_io as iio
    dat = _write_dat(str(tmp_path / "s.dat"), [
        "#@INSexp " + " ".join(str(float(v)) for v in _LEGACY_INS),
        f"#@INSint {_LEGACY_MULCO} {_LEGACY_X0}",
    ])
    meta = iio.parse_dat_instrumental_metadata(dat)
    assert meta['has_insexp'] and not meta['has_insacc']
    assert meta['INSacc'] is None


def test_a_gaussian_payload_on_the_accurate_marker_is_ignored(tmp_path):
    """Defensive: #@INSacc must carry a tagged theoretical function or nothing."""
    from syncmoss import instrumental_io as iio
    dat = _write_dat(str(tmp_path / "s.dat"), [
        "#@INSexp " + " ".join(str(float(v)) for v in _LEGACY_INS),
        f"#@INSint {_LEGACY_MULCO} {_LEGACY_X0}",
        "#@INSacc 0.1 0.0 1.0",
    ])
    meta = iio.parse_dat_instrumental_metadata(dat)
    assert not meta['has_insacc']
    assert meta['INSacc'] is None
