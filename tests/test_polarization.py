"""Incident-beam polarization readout of the polarized ("thick") models.

Covers the two knobs added on top of the fully-polarized SMS assumption:

  (1) ``models.SMS_LINEAR_POLARIZATION`` -- the SMS beam may be partially
      polarized. The transmission is read out of the density matrix
      ``rho = diag((1+p)/2, (1-p)/2)``, so the spectrum must be the linear mix
      ``(1+p)/2 * [read (1,1)] + (1-p)/2 * [read (2,2)]``. p = 1 (the default)
      reproduces the original (1,1)-only readout.

  (2) A conventional radioactive source (CMS, ``Met == 1``) is unpolarized:
      ``rho = I/2`` -> half the trace of the transmission matrix, a FIXED 1:1
      mixture that ignores ``SMS_LINEAR_POLARIZATION`` entirely. This equals an
      SMS beam at p = 0, which is what makes thick models valid for a CMS source.

Pure NumPy/Numba path (``TImod`` called directly, single process) -- no Qt and
no multiprocessing pool, so monkeypatching the module constant takes effect.
"""
import numpy as np
import pytest

import syncmoss.models as m5

_E = np.linspace(-10.0, 10.0, 96)

# Anisotropic, genuinely thick components (off-axis orientation so the two
# polarization eigenchannels differ, i.e. [expm(-Sigma)]_11 != _22).
_THICK_CASES = {
    "Doublet_(thick)":     [20.0, 0.0, 1.0, 0.098, 0.15, 50.0, 30.0, 1.0, 1.0],
    "Sextet_(thick)":      [20.0, 0.0, 0.0, 33.0, 0.098, 0.15, 50.0, 30.0, 1.0, 0.0, 0.0, 0.0, 3.0],
    "Hamilton_mc_(thick)": [20.0, 0.0, 0.4, 33.0, 0.098, 0.15, 0.0, 20.0, 30.0, 40.0, 25.0, 35.0],
}


def _compute(model, params, mett, pol, monkeypatch):
    """Per-energy absorber transmission C_a(E) for one thick component.

    ``mett`` selects the source (0 = SMS, 1 = CMS); ``pol`` sets
    ``SMS_LINEAR_POLARIZATION`` for this call.
    """
    monkeypatch.setattr(m5, "SMS_LINEAR_POLARIZATION", pol)
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array([model]), _E, 0.0, 1.0,
                 np.array([]), [], [], Met=-1, Mett=mett),
        dtype=float,
    )


@pytest.mark.parametrize("model", sorted(_THICK_CASES))
def test_sms_default_polarization_reads_11_element(model, monkeypatch):
    """p = 1 must reproduce the original (1,1)-only SMS readout."""
    params = _THICK_CASES[model]
    # The realistic-beam default value shipped in the module is 0.98 ...
    assert m5.SMS_LINEAR_POLARIZATION == 0.98
    # ... and computing at p = 1 is a plain (1,1) readout, in (0, 1].
    y = _compute(model, params, mett=0, pol=1.0, monkeypatch=monkeypatch)
    assert np.all(np.isfinite(y))
    assert np.all(y > 0.0) and np.all(y <= 1.0 + 1e-9)


@pytest.mark.parametrize("model", sorted(_THICK_CASES))
def test_sms_partial_polarization_is_density_matrix_mix(model, monkeypatch):
    """C_a(p) == (1+p)/2 * C_a(read 11) + (1-p)/2 * C_a(read 22)."""
    params = _THICK_CASES[model]
    y_read11 = _compute(model, params, mett=0, pol=1.0, monkeypatch=monkeypatch)   # rho = diag(1,0)
    y_read22 = _compute(model, params, mett=0, pol=-1.0, monkeypatch=monkeypatch)  # rho = diag(0,1)
    # Off-axis anisotropy: the two eigenchannels really do differ.
    assert np.max(np.abs(y_read11 - y_read22)) > 1e-3
    for p in (0.0, 0.3, 0.7, 0.98):
        y = _compute(model, params, mett=0, pol=p, monkeypatch=monkeypatch)
        expected = 0.5 * (1.0 + p) * y_read11 + 0.5 * (1.0 - p) * y_read22
        assert np.allclose(y, expected, rtol=1e-9, atol=1e-11), (
            f"{model}: partial-polarization readout is not the density-matrix mix at p={p}"
        )


@pytest.mark.parametrize("model", sorted(_THICK_CASES))
def test_cms_is_fixed_half_trace(model, monkeypatch):
    """CMS (Met==1) reads half the trace, equals SMS at p=0, and ignores the knob."""
    params = _THICK_CASES[model]
    y_sms_unpol = _compute(model, params, mett=0, pol=0.0, monkeypatch=monkeypatch)   # SMS p=0
    y_cms = _compute(model, params, mett=1, pol=1.0, monkeypatch=monkeypatch)         # CMS, knob "fully polarized"
    y_cms_other = _compute(model, params, mett=1, pol=0.3, monkeypatch=monkeypatch)   # CMS, other knob value

    assert np.all(np.isfinite(y_cms))
    assert np.all(y_cms > 0.0) and np.all(y_cms <= 1.0 + 1e-9)
    # CMS is a fixed 1:1 mixture -> unaffected by SMS_LINEAR_POLARIZATION.
    assert np.allclose(y_cms, y_cms_other, rtol=1e-12, atol=1e-13), f"{model}: CMS must ignore the SMS knob"
    # ... and it equals the unpolarized SMS limit (both are the half-trace).
    assert np.allclose(y_cms, y_sms_unpol, rtol=1e-9, atol=1e-11), f"{model}: CMS != SMS(p=0)"
    # ... and genuinely differs from the fully-polarized SMS readout.
    y_sms_pol = _compute(model, params, mett=0, pol=1.0, monkeypatch=monkeypatch)
    assert np.max(np.abs(y_cms - y_sms_pol)) > 1e-3, f"{model}: CMS should differ from fully-polarized SMS"


def test_cms_readout_independent_of_azimuth(monkeypatch):
    """For the matrix-based thick models an unpolarized (CMS) source cannot sense
    the azimuth phi_h: the half-trace depends only on the polar angle theta_h.

    (This is the azimuthal-immateriality of an unpolarized source. It does NOT
    extend to Hamilton_mc_(thick)'s alpha_k, which re-aims the beam k about h in
    this parametrization and so does affect the readout -- a "nonsense" parameter
    for a CMS spectrum that is left to the user.)"""
    # Sextet_(thick): columns 6, 7 are theta_h, phi_h.
    base = list(_THICK_CASES["Sextet_(thick)"])
    ref = _compute("Sextet_(thick)", base, mett=1, pol=1.0, monkeypatch=monkeypatch)
    for phi_h in (0.0, 90.0, 200.0, -45.0):
        p = list(base)
        p[7] = phi_h
        y = _compute("Sextet_(thick)", p, mett=1, pol=1.0, monkeypatch=monkeypatch)
        assert np.allclose(y, ref, rtol=1e-9, atol=1e-11), (
            f"CMS readout must not depend on the azimuth phi_h (deviated at phi_h={phi_h})"
        )


def test_scalar_model_unaffected_by_polarization(monkeypatch):
    """A non-thick model has no polarization matrix, so the knob and the CMS
    branch leave it untouched (regression guard for the readout change)."""
    params = [1.0, 0.0, 0.0, 33.0, 0.098, 0.1, 0.5, 0.0, 0.0, 0.0, 3.0]  # scalar Sextet
    y_ref = _compute("Sextet", params, mett=0, pol=1.0, monkeypatch=monkeypatch)
    for mett, pol in ((0, 0.0), (0, 0.5), (1, 1.0)):
        y = _compute("Sextet", params, mett=mett, pol=pol, monkeypatch=monkeypatch)
        assert np.array_equal(y, y_ref), "scalar model must not depend on polarization/source readout"
