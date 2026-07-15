"""Fast unit tests for the pure helper functions of ``syncmoss.Calibration``.

The end-to-end calibration fits are covered by ``test_calibration.py`` (slow);
these tests exercise the extracted, fit-free pieces -- raw-file parsing for
every supported format, phase detection, minimum finding, folding and the
``Calibration.dat`` writer -- so a regression there is caught in milliseconds
instead of a multi-minute full fit.
"""
import os

import numpy as np
import pytest

from syncmoss.Calibration import (
    _deepest_line_channels,
    _detect_phase_offset,
    _fold_sinusoidal,
    _fold_triangular,
    _load_raw_counts,
    _parabola,
    _subtract_distortion,
    _write_calibration_dat,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CMS_MCA = os.path.join(_HERE, "data", "synthetic_alpha_fe_cms_linear.mca")
_SMS_MCA = os.path.join(_HERE, "data", "alpha_fe_sms_000.mca")


# --------------------------------------------------------------------------- #
# _load_raw_counts                                                            #
# --------------------------------------------------------------------------- #

def test_load_mca_synthetic_fixture():
    counts = _load_raw_counts(_CMS_MCA)
    assert counts.ndim == 1
    assert counts.size == 256
    assert np.all(counts > 0)


def test_load_mca_real_fixture():
    if not os.path.exists(_SMS_MCA):
        pytest.skip("SMS fixture missing")
    counts = _load_raw_counts(_SMS_MCA)
    assert counts.size % 2 == 0  # loader guarantees an even channel count for .mca
    assert np.all(np.isfinite(counts))


def test_load_ws5(tmp_path):
    values = [100, 200, 300, 400]
    p = tmp_path / "spec.ws5"
    p.write_text("# comment\n<header>\n" + "\n".join(str(v) for v in values) + "\n")
    assert _load_raw_counts(str(p)).tolist() == values


def test_load_m1_skips_first_data_row(tmp_path):
    # .m1: counts live in column 5; the first data row is a header and skipped.
    rows = ["0 0 0 0 999"] + [f"0 0 0 0 {v}" for v in (10, 20, 30)]
    p = tmp_path / "spec.m1"
    p.write_text("\n".join(rows) + "\n")
    assert _load_raw_counts(str(p)).tolist() == [10, 20, 30]


def test_load_moe_skips_float_header_lines(tmp_path):
    p = tmp_path / "spec.moe"
    p.write_text("2.5\n100\n200\n300\n")  # '2.5' is a header (contains '.')
    assert _load_raw_counts(str(p)).tolist() == [100, 200, 300]


def test_load_mcs_binary(tmp_path):
    values = np.array([7, 8, 9, 10], dtype=np.uint32)
    p = tmp_path / "spec.mcs"
    p.write_bytes(b"\x00" * 256 + values.tobytes())
    assert _load_raw_counts(str(p)).tolist() == [7, 8, 9, 10]


def test_load_unknown_extension_raises(tmp_path):
    p = tmp_path / "spec.xyz"
    p.write_text("1\n2\n")
    with pytest.raises(ValueError, match="unsupported calibration file type"):
        _load_raw_counts(str(p))


# --------------------------------------------------------------------------- #
# _detect_phase_offset                                                        #
# --------------------------------------------------------------------------- #

def test_phase_zero_for_perfect_mirror():
    rng = np.random.default_rng(1)
    half1 = rng.uniform(100, 200, 64)
    assert _detect_phase_offset(half1, half1[::-1]) == 0


def test_phase_detects_known_shift():
    # Craft halves that mirror-match exactly only when the first 3 channels
    # are discarded: half1[3:] == half2[3:][::-1]. The detector reports the
    # offset in full-spectrum channels: -(-3)*2 = 6.
    rng = np.random.default_rng(2)
    base = rng.uniform(100, 200, 29)
    half1 = np.concatenate((rng.uniform(300, 400, 3), base))
    half2 = np.concatenate((rng.uniform(500, 600, 3), base[::-1]))
    assert _detect_phase_offset(half1, half2) == 6


# --------------------------------------------------------------------------- #
# _deepest_line_channels                                                      #
# --------------------------------------------------------------------------- #

def test_deepest_line_channels():
    counts = np.full(128, 1000.0)
    counts[10] = 100.0   # deepest in first quarter (channels 0..31)
    counts[40] = 200.0   # deepest in second quarter (channels 32..63)
    assert _deepest_line_channels(counts) == (10, 40)


def test_deepest_line_channels_prefers_first_occurrence_inside_quarter():
    counts = np.full(128, 1000.0)
    counts[[5, 20]] = 100.0    # duplicated minimum in the first quarter
    counts[[35, 50]] = 200.0   # duplicated minimum in the second quarter
    assert _deepest_line_channels(counts) == (5, 35)


# --------------------------------------------------------------------------- #
# distortion helpers                                                          #
# --------------------------------------------------------------------------- #

def test_subtract_distortion_recovers_flat_signal():
    n = 64
    channels = np.arange(n, dtype=float)
    flat = np.full(n, 5000.0)
    # pS layout: [4..6] parabola of half 1 (centre, curvature, offset),
    # [7..8] parabola of half 2 (centre, curvature).
    pS = np.zeros(9)
    pS[4:7] = [10.0, 0.5, 30.0]
    pS[7:9] = [40.0, -0.25]
    counts = flat.copy()
    counts[:32] += _parabola(channels[:32], [pS[4], pS[5], pS[6]])
    counts[32:] += _parabola(channels[32:], [pS[7], pS[8], 0.0])
    np.testing.assert_allclose(_subtract_distortion(counts, channels, pS), flat)


# --------------------------------------------------------------------------- #
# folding                                                                     #
# --------------------------------------------------------------------------- #

def test_fold_triangular_symmetric():
    # Symmetric ramps (pS[0] == pS[1]) -> the whole spectrum folds pairwise.
    ramp = np.linspace(6.0, -6.0, 8)
    v_axis = np.concatenate((ramp, ramp[::-1]))
    counts = np.concatenate((np.arange(8.0), np.arange(8.0)[::-1]))
    fit = counts + 1.0
    pS = np.array([6.0, 6.0, 1.0])
    v_f, data_f, fit_f, resid, n1, n2 = _fold_triangular(v_axis, counts, fit, pS)
    assert (n1, n2) == (0, 15)
    assert v_f.size == data_f.size == fit_f.size == 8
    np.testing.assert_allclose(v_f, ramp)              # both visits at the same v
    np.testing.assert_allclose(data_f, 2 * np.arange(8.0))
    np.testing.assert_allclose(resid, 0.0)             # symmetric data -> no residual


def test_fold_sinusoidal_symmetric():
    # Axis whose second half exactly mirrors the first and turnarounds at the
    # ends -> single central fold segment, zero residual for symmetric data.
    a = np.linspace(5.0, -5.0, 8)
    v_axis = np.concatenate((a, a[::-1]))
    counts = np.concatenate((np.arange(8.0), np.arange(8.0)[::-1]))
    fit = counts * 1.5
    v_f, data_f, fit_f, resid, n1, n2 = _fold_sinusoidal(v_axis, counts, fit, a[0])
    assert (n1, n2) == (0, 15)
    assert v_f.size == data_f.size == fit_f.size
    np.testing.assert_allclose(data_f, 2 * np.arange(8.0))
    np.testing.assert_allclose(resid, 0.0)


# --------------------------------------------------------------------------- #
# Calibration.dat writer                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,tag", [(0, "sin"), (1, "lin")])
def test_write_calibration_dat(tmp_path, method, tag):
    v = np.array([-1.5, 0.0, 1.5])
    c = np.array([100.0, 90.0, 110.0])
    _write_calibration_dat(str(tmp_path), method, 3, 508, v, c)
    lines = (tmp_path / "Calibration.dat").read_text().splitlines()
    header = lines[0].split("\t")
    assert header == ["#", f"{tag} ", "3", "508"]
    data = np.array([ln.split("\t") for ln in lines[1:4]], dtype=float)
    np.testing.assert_allclose(data[:, 0], v)
    np.testing.assert_allclose(data[:, 1], c)
