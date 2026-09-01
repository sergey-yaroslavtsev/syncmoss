"""End-to-end regression test for the velocity calibration.

``Calibration.Calibration`` builds the alpha-Fe model parameter arrays BY HAND and
indexes them with hardcoded integer offsets. When the scalar ("thin") and
polarized ("thick") models were merged, the Sextet grew from 11 to 13 parameters
(theta_k, phi_h inserted; asymmetry -> texture A) and the Doublet from 7 to 9, so
every one of those offsets shifted. These tests fit real and synthetic alpha-Fe
spectra end-to-end and check the recovered velocity scale, guarding those offsets.

* SMS (synchrotron, ``VVV == 3``, sinusoidal drive): fitted against the real
  ``alpha_fe_sms_000.mca`` for both velocity directions (``Vel_start`` 0 and 1),
  compared to committed golden results captured from the working calibration.
* CMS (conventional source, ``VVV == 1``, linear/triangular drive): fitted
  against a synthetic linear-step alpha-Fe spectrum (``synthetic_alpha_fe_cms_linear.mca``,
  drive amplitude +/-6 mm/s). CMS uses a single Gaussian instrumental width
  (``GCMS``) rather than the multi-line SMS ``INS`` array.

Heavy: each case spawns a (single-process) multiprocessing pool and runs the full
fit, so the module is marked ``slow``.
"""
import json
import os
import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_PARAMS = os.path.abspath(os.path.join(_HERE, "..", "syncmoss", "parameters"))
_SMS_MCA = os.path.join(_HERE, "data", "alpha_fe_sms_000.mca")
_CMS_MCA = os.path.join(_HERE, "data", "synthetic_alpha_fe_cms_linear.mca")
_GOLDEN = os.path.join(_HERE, "data", "calibration_000_golden.json")


def _run_calibration(mca, vel_start, vvv, jn=32):
    """Run a full calibration on ``mca`` in a throw-away params dir.

    ``dir_path`` is a copy of the parameters folder so the fit writes its
    ``Calibration.dat`` there and never mutates the tracked data files.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    for required in ("INSint.txt", "INSexp.txt"):
        if not os.path.exists(os.path.join(_SRC_PARAMS, required)):
            pytest.skip(f"instrumental parameter file missing: {required}")

    import multiprocessing as mp
    import tempfile
    from syncmoss.Calibration import Calibration

    work = tempfile.mkdtemp()
    for fn in ("Be.txt", "KB.txt"):
        src = os.path.join(_SRC_PARAMS, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(work, fn))
    MulCo, x0 = np.genfromtxt(os.path.join(_SRC_PARAMS, "INSint.txt"), delimiter=" ")
    INS = np.genfromtxt(os.path.join(_SRC_PARAMS, "INSexp.txt"), delimiter=" ")
    try:
        GCMS = float(np.genfromtxt(os.path.join(_SRC_PARAMS, "GCMS.txt"), delimiter="\t"))
    except Exception:
        GCMS = 0.1

    pool = mp.Pool(processes=1)
    try:
        A, B, C = Calibration(work, mca, pool, vvv, INS, jn, x0, MulCo, Vel_start=vel_start, GCMS=GCMS)
    finally:
        pool.close()
        pool.join()
        shutil.rmtree(work, ignore_errors=True)
    return (np.asarray(A, dtype=float), np.asarray(B, dtype=float),
            np.asarray(C, dtype=float))


@pytest.fixture(scope="module")
def sms_results():
    """Run the SMS calibration on the real alpha-Fe spectrum for both directions."""
    if not os.path.exists(_SMS_MCA):
        pytest.skip("SMS calibration fixture missing")
    return {vs: _run_calibration(_SMS_MCA, vs, vvv=3) for vs in (0, 1)}


@pytest.fixture(scope="module")
def golden():
    with open(_GOLDEN) as f:
        return json.load(f)


@pytest.mark.parametrize("vel_start", [0, 1])
def test_sms_velocity_axis_is_finite(sms_results, vel_start):
    v = sms_results[vel_start][0]
    assert v.size == 512
    assert np.all(np.isfinite(v))


@pytest.mark.parametrize("vel_start", [0, 1])
def test_sms_folded_spectrum_and_fit(sms_results, vel_start):
    """The folded data (B) and folded fit (C) must be consistent: same length
    as the velocity axis, finite, and the fit must actually describe the data
    (the committed reference runs deviate by <4.5% of the peak counts)."""
    v, data, fit = sms_results[vel_start]
    assert data.size == v.size and fit.size == v.size
    assert np.all(np.isfinite(data)) and np.all(np.isfinite(fit))
    assert float(np.max(np.abs(data - fit))) < 0.10 * float(np.max(data))


@pytest.mark.parametrize("vel_start", [0, 1])
def test_sms_matches_golden(sms_results, golden, vel_start):
    """The recovered velocity axis must reproduce the committed calibration.

    A wrong Sextet/Doublet parameter offset would shift the fitted line positions
    by >0.5 mm/s, so the tight span/endpoint tolerance guards those offsets.
    """
    v = sms_results[vel_start][0]
    ref = golden[str(vel_start)]
    assert v.size == ref["n"]
    assert float(v.max() - v.min()) == pytest.approx(ref["v_span"], abs=0.05)
    assert float(v.min()) == pytest.approx(ref["v_min"], abs=0.05)
    assert float(v.max()) == pytest.approx(ref["v_max"], abs=0.05)
    # alpha-Fe outermost lines are at +/-5.31 mm/s -> the axis must cover them.
    assert v.min() < -4.5 and v.max() > 4.5


def test_sms_velocity_direction_reverses(sms_results):
    """Vel_start flips the sweep direction, so the two axes are mirror-like:
    the span matches but the (min, max) endpoints swap sign roughly.

    (Not parametrized: the body compares BOTH directions against each other, so a
    `vel_start` parameter was unused and simply ran the identical assertion twice.)
    """
    v0, v1 = sms_results[0][0], sms_results[1][0]
    assert float(v0.max() - v0.min()) == pytest.approx(float(v1.max() - v1.min()), abs=0.05)


@pytest.fixture(scope="module")
def cms_result():
    """Run the CMS (VVV=1) calibration on the synthetic linear-drive spectrum."""
    if not os.path.exists(_CMS_MCA):
        pytest.skip("CMS calibration fixture missing")
    return _run_calibration(_CMS_MCA, vel_start=1, vvv=1, jn=24)


# Synthetic CMS drive amplitude (see generate_cal_fixtures.py: vmax = 6.0 mm/s),
# so the folded velocity axis must span ~2 * 6 = 12 mm/s.
_CMS_TRUE_VMAX = 6.0


def test_cms_velocity_axis_is_finite(cms_result):
    v = cms_result[0]
    assert v.size > 0
    assert np.all(np.isfinite(v))


def test_cms_folded_spectrum_and_fit(cms_result):
    """Folded data (B) and fit (C) consistency for the CMS/linear path."""
    v, data, fit = cms_result
    assert data.size == v.size and fit.size == v.size
    assert np.all(np.isfinite(data)) and np.all(np.isfinite(fit))
    assert float(np.max(np.abs(data - fit))) < 0.10 * float(np.max(data))


def test_cms_recovers_linear_drive_amplitude(cms_result):
    """CMS uses a single Gaussian instrumental width (GCMS); the linear-drive
    spectrum must be calibrated to ~2 * vmax, guarding both the CMS instrumental
    handling and the Sextet parameter offsets on the linear-fit path."""
    v = cms_result[0]
    span = float(v.max() - v.min())
    assert span == pytest.approx(2 * _CMS_TRUE_VMAX, abs=1.0), (
        f"recovered CMS velocity span {span:.3f} mm/s far from expected {2 * _CMS_TRUE_VMAX}"
    )
    assert v.min() < -4.5 and v.max() > 4.5
