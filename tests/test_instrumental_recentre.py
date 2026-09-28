"""Calibration moves the axis; the instrumental function must move with it.

Calibration relabels the velocity axis by ``INS_shift = ins_centroid(INS)``.
That is a GAUGE transformation -- for

    T(v) = int dE S(E) exp(-sigma(E + v))

replacing S(E) by S(E - d) gives exactly T(v + d) -- so moving the axis is only
correct if the source moves with it. Doing just the axis half leaves every later
fit off by the centroid, which for the simulated source is -0.0144 mm/s, about
15 % of a natural linewidth, and is what forced a sextet's shift open.

Doing both halves is the whole point: the fit is unchanged, the gravity centre
lands at zero, and the NEXT calibration has nothing left to do.
"""
import os

import numpy as np
import pytest

import syncmoss.instrumental_io as io
import syncmoss.sms_theory as smst

from conftest import redirect_params_dir_to_tmp


@pytest.fixture
def app_with_theory_ins(physics_app, tmp_path):
    redirect_params_dir_to_tmp(physics_app, tmp_path)
    io.write_accurate_instrumental(physics_app, io.default_theory_instrumental())
    return physics_app


def test_the_default_is_not_already_centred(app_with_theory_ins):
    """Otherwise the rest of this file would pass vacuously."""
    INS = io.read_accurate_instrumental(app_with_theory_ins)
    assert abs(smst.ins_centroid(INS)) > 0.005


def test_recentring_zeroes_the_gravity_centre(app_with_theory_ins):
    moved = io.recentre_instrumental_after_calibration(app_with_theory_ins)
    INS = io.read_accurate_instrumental(app_with_theory_ins)
    assert smst.ins_centroid(INS) == pytest.approx(0.0, abs=1e-9)
    assert moved == pytest.approx(-0.0144, abs=2e-3)


def test_it_moves_only_the_shift(app_with_theory_ins):
    """A gauge transformation: nothing physical may change."""
    before = smst.decode_physical(io.read_accurate_instrumental(app_with_theory_ins))
    moved = io.recentre_instrumental_after_calibration(app_with_theory_ins)
    after = smst.decode_physical(io.read_accurate_instrumental(app_with_theory_ins))
    assert after['shift'] == pytest.approx(before['shift'] - moved, abs=1e-12)
    for k in ('theta_urad', 'B_s', 'dEQ', 'gauss_fwhm', 'mosaic_urad', 'f_LM'):
        assert after[k] == pytest.approx(before[k]), k


def test_the_shape_is_only_translated(app_with_theory_ins):
    """The fit must be untouched: the new shape is the old one moved by -moved."""
    v = np.linspace(-1.5, 1.5, 6001)
    before = io.read_accurate_instrumental(app_with_theory_ins)
    S0 = smst.ins_shape(before, v)
    moved = io.recentre_instrumental_after_calibration(app_with_theory_ins)
    S1 = smst.ins_shape(io.read_accurate_instrumental(app_with_theory_ins), v)
    translated = np.interp(v + moved, v, S0, left=0.0, right=0.0)
    inner = (v > -1.0) & (v < 1.0)
    assert np.max(np.abs(S1[inner] - translated[inner])) < 1e-3 * S0.max()


def test_it_is_a_fixed_point(app_with_theory_ins):
    """A second calibration of an unchanged setup must do nothing at all."""
    io.recentre_instrumental_after_calibration(app_with_theory_ins)
    stored = np.array(io.read_accurate_instrumental(app_with_theory_ins))
    assert io.recentre_instrumental_after_calibration(app_with_theory_ins) == 0.0
    again = np.array(io.read_accurate_instrumental(app_with_theory_ins))
    assert np.array_equal(stored, again), "the file was rewritten for nothing"


def _first_moment(a):
    return sum(a[i * 3 + 1] * a[i * 3 + 2] ** 2 for i in range(len(a) // 3))


@pytest.fixture
def app_with_offcentre_gaussians(physics_app, tmp_path):
    """A Gaussian sum deliberately off centre, as a fresh search could give."""
    redirect_params_dir_to_tmp(physics_app, tmp_path)
    io.write_accurate_instrumental(physics_app, None)      # Gaussian selected
    g = np.array(io.get_legacy_sms_instrumental(physics_app), dtype=float)
    g[1::3] += 0.05
    with open(os.path.join(physics_app.params_dir, io.INS_EXP_FILE), "w") as f:
        f.write(' '.join(str(float(v)) for v in g) + ' ')
    return physics_app


def test_a_gaussian_sum_is_recentred_too(app_with_offcentre_gaussians):
    """The argument is about the GAUGE, not about which shape is used. The
    shipped sum's centroid is small by accident (+0.0008 mm/s, because it was
    fitted against an axis that had already absorbed it), not by construction.
    """
    app = app_with_offcentre_gaussians
    before = np.array(io.get_legacy_sms_instrumental(app), dtype=float)
    assert abs(_first_moment(before)) > 0.01, "the fixture is not off centre"

    moved = io.recentre_instrumental_after_calibration(app)
    after = np.array(io.get_legacy_sms_instrumental(app), dtype=float)

    assert _first_moment(after) == pytest.approx(0.0, abs=1e-9)
    assert moved == pytest.approx(_first_moment(before), abs=1e-9)
    # a translation: every position moved by the same amount, widths and
    # amplitudes untouched
    assert np.allclose(after[1::3], before[1::3] - moved, atol=1e-12)
    assert np.allclose(after[0::3], before[0::3], atol=1e-12)
    assert np.allclose(after[2::3], before[2::3], atol=1e-12)


def test_gaussian_recentring_is_a_fixed_point(app_with_offcentre_gaussians):
    app = app_with_offcentre_gaussians
    io.recentre_instrumental_after_calibration(app)
    stored = np.array(io.get_legacy_sms_instrumental(app))
    assert io.recentre_instrumental_after_calibration(app) == 0.0
    assert np.array_equal(stored, np.array(io.get_legacy_sms_instrumental(app)))
