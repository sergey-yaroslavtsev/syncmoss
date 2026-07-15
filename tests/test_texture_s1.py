"""The magnetic polar-order texture parameter A_m (S1) of the Faraday-active
polarized models (Sextet, MDGD, Relax_2S).

A_m in [-1, 1] scales the resolved sigma+- Faraday term via
S1 = A_m * sqrt((1 + 2A) / 3) (models._texture_s1), where A is the ordinary
uniaxial texture order parameter. These tests pin the physics of that term:

  * it is bounded by Cauchy-Schwarz (the sqrt is the |S1| ceiling);
  * it is a purely thick, off-axis observable -- it vanishes for the axis along
    the beam-perpendicular readout is unaffected in the thin limit, and it does
    nothing when the axis lies in the polarization plane (theta_k = 90, n_z = 0);
  * the SIGN of A_m is observable for a polarized (SMS) beam once the dispersive
    (Kramers--Kronig) part is included -- S1 -> -S1 is no longer a pure complex
    conjugation of the non-Hermitian cross-section -- but stays invisible to an
    unpolarized (CMS, half-trace) beam;
  * A_m = 0 (the default) restores the exact random-powder limit at A = 0.

Pure NumPy/Numba path (TImod called directly), like test_polarization.
"""
import numpy as np
import pytest

import syncmoss.models as m5

_E = np.linspace(-10.0, 10.0, 96)

# For each Faraday-active model: a thick, single-crystal (A=1) parameter template
# with the A_m slot last, plus the indices of theta_k, the texture A, and A_m.
_MODELS = {
    # A_m sits immediately after A. Tuple: (base params, theta_k idx, A idx, A_m idx).
    #                T    d    e     H     L      G   thk   phh   A   A_m  a+   a-  GH   I13   thidx aidx amidx
    "Sextet":   ([20.0, 0.0, 0.0, 33.0, 0.098, 0.15, 40.0, 30.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0], 6, 8, 9),
    "MDGD":     ([20.0, 0.0, 0.0, 33.0, 0.098, 0.15, 0.3, 0.1, 0.1, 0.1, 40.0, 30.0, 1.0, 0.0, 0.0, 0.0, 3.0], 10, 12, 13),
    # Relax_2S: two states with the SAME-sign field (both +33 T) -> a net-magnetised
    # (not mirror-symmetric) system, so the sigma+- Faraday term does not cancel and
    # A_m genuinely matters. (A mirror +-33 T pair at equal population is unmagnetised
    # and would zero the Faraday term whatever A_m is -- correct physics, but it would
    # not exercise A_m.) Layout: T,d1,e1,H1,d2,e2,H2,L,thk,phh,A,A_m,W12,P1/P2.
    "Relax_2S": ([20.0, 0.0, 0.0, 33.0, 0.3, 0.0, 33.0, 0.12, 40.0, 30.0, 1.0, 0.0, 0.3, 1.0], 8, 10, 11),
}
_FARADAY_MODELS = sorted(_MODELS)


def _compute(model, params, mett=0):
    # Pin the complex-Voigt method/sign to the shipping defaults so these tests
    # are independent of any developer switch of models.COMPLEX_VOIGT_METHOD.
    m5.COMPLEX_VOIGT_METHOD = 'pseudo'
    m5.DISPERSION_SIGN = +1.0
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array([model]), _E, 0.0, 1.0,
                 np.array([]), [], [], Met=-1, Mett=mett),
        dtype=float,
    )


def _spec(model, am=0.0, theta=None, a_tex=None, thickness=None, mett=0):
    base, thidx, aidx, amidx = _MODELS[model]
    p = list(base)
    p[amidx] = am
    if theta is not None:
        p[thidx] = theta
    if a_tex is not None:
        p[aidx] = a_tex
    if thickness is not None:
        p[0] = thickness
    return _compute(model, p, mett=mett)


# --- the S1 = A_m * sqrt((1 + 2A)/3) parametrisation itself -------------------

@pytest.mark.parametrize("A", [-0.5, -0.25, 0.0, 0.3, 0.5, 1.0])
def test_s1_is_fraction_of_cauchy_schwarz_bound(A):
    """A_m is S1 as a fraction of its Cauchy-Schwarz ceiling sqrt((1+2A)/3), so
    A_m = +-1 sits exactly on the bound and A_m = 0 gives S1 = 0."""
    bound = np.sqrt((1.0 + 2.0 * A) / 3.0)
    assert m5._texture_s1(A, 1.0) == pytest.approx(bound)
    assert m5._texture_s1(A, -1.0) == pytest.approx(-bound)
    assert m5._texture_s1(A, 0.0) == 0.0
    for am in (-0.7, 0.4):
        assert m5._texture_s1(A, am) == pytest.approx(am * bound)


def test_s1_single_crystal_is_unity():
    """A = A_m = 1 -> S1 = 1, the fully-magnetised single crystal."""
    assert m5._texture_s1(1.0, 1.0) == pytest.approx(1.0)


def test_s1_clamped_at_planar_texture():
    """At A = -1/2 (perfect planar texture) the bound is 0, so S1 is forced to 0
    for any A_m; the sqrt argument is clamped so it never goes imaginary."""
    assert m5._texture_s1(-0.5, 1.0) == 0.0
    assert m5._texture_s1(-0.6, 1.0) == 0.0   # below the valid range -> clamped, not NaN


# --- the term is a thick, off-axis observable ---------------------------------

@pytest.mark.parametrize("model", _FARADAY_MODELS)
def test_am_has_no_effect_when_axis_in_polarization_plane(model):
    """theta_k = 90 puts the axis in the polarization plane (n_z = 0), so the
    Faraday term (proportional to n_z) vanishes and A_m does nothing."""
    y0 = _spec(model, am=0.0, theta=90.0)
    y1 = _spec(model, am=1.0, theta=90.0)
    assert np.allclose(y0, y1, rtol=1e-12, atol=1e-13), (
        f"{model}: A_m must not affect the spectrum at theta_k = 90 (n_z = 0)"
    )


@pytest.mark.parametrize("model", _FARADAY_MODELS)
def test_am_changes_offaxis_thick_spectrum(model):
    """For a thick, off-axis (theta_k = 40) single crystal, turning on the Faraday
    term (A_m 0 -> 1) genuinely moves the spectrum -- so A_m is wired in."""
    y0 = _spec(model, am=0.0, theta=40.0)
    y1 = _spec(model, am=1.0, theta=40.0)
    assert np.max(np.abs(y0 - y1)) > 1e-3, (
        f"{model}: A_m should change an off-axis thick spectrum"
    )


@pytest.mark.parametrize("model", _FARADAY_MODELS)
def test_am_sign_observable_for_sms_not_cms(model):
    """The sign of A_m and the beam polarization.

    In the absorption-only (Hermitian) model reversing A_m simply conjugated the
    cross-section, so only |S1| was ever measurable. With the dispersive
    (Kramers--Kronig) completion the cross-section is complex non-Hermitian and
    S1 -> -S1 is no longer a pure conjugation, so the SIGN of A_m becomes a real
    observable -- but only for a POLARIZED (SMS, Mett=0) beam. An UNPOLARIZED
    (CMS, Mett=1) beam is read out as the half-trace, which stays invariant under
    conjugation, so it still cannot see the sign (an unpolarized beam has no
    handedness to couple to the magnetisation direction)."""
    sms_plus = _spec(model, am=0.7, theta=40.0, mett=0)
    sms_minus = _spec(model, am=-0.7, theta=40.0, mett=0)
    assert np.max(np.abs(sms_plus - sms_minus)) > 1e-3, (
        f"{model}: with dispersion the sign of A_m must be observable for a polarized SMS beam"
    )
    cms_plus = _spec(model, am=0.7, theta=40.0, mett=1)
    cms_minus = _spec(model, am=-0.7, theta=40.0, mett=1)
    assert np.allclose(cms_plus, cms_minus, rtol=1e-9, atol=1e-11), (
        f"{model}: the sign of A_m must NOT be observable for an unpolarized CMS beam"
    )


@pytest.mark.parametrize("model", _FARADAY_MODELS)
def test_am_vanishes_in_thin_limit(model):
    """In the thin limit the Faraday term is off-diagonal and traceless, so A_m
    affects neither the (1,1) nor the half-trace readout: a very thin absorber is
    (nearly) independent of A_m, unlike the thick case above."""
    thin0 = _spec(model, am=0.0, theta=40.0, thickness=0.02)
    thin1 = _spec(model, am=1.0, theta=40.0, thickness=0.02)
    assert np.max(np.abs(thin0 - thin1)) < 1e-5, (
        f"{model}: A_m must not affect the thin-limit spectrum"
    )


@pytest.mark.parametrize("model", _FARADAY_MODELS)
def test_powder_limit_is_orientation_independent(model):
    """At A = 0 AND A_m = 0 every building block averages to the identity, so the
    component is an exact random powder -- its spectrum cannot depend on the axis
    orientation (neither theta_k nor, implicitly, the Faraday term)."""
    ref = _spec(model, am=0.0, a_tex=0.0, theta=40.0)
    for theta in (0.0, 90.0, 123.0):
        y = _spec(model, am=0.0, a_tex=0.0, theta=theta)
        assert np.allclose(y, ref, rtol=1e-11, atol=1e-13), (
            f"{model}: a random powder (A=0, A_m=0) must not depend on theta_k={theta}"
        )
