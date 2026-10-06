"""What a fit with exclusion regions hands the minimiser, and what it hands back.

The minimiser is replaced by one that returns the start vector, so no time is
spent fitting; everything around it runs for real (serial pool). The spectrum
is the tests' frozen calibration spectrum, written to a .dat with the precision
"Save spectrum without excluded points" writes, so the saved file and the
regions can be compared point for point.
"""
import os

import numpy as np
import pytest

from syncmoss import exclusion_regions as er
from syncmoss import fitting_io, one_model
from syncmoss.spectrum_io import load_spectrum, save_spectrum_with_metadata, process_without_excluded

from conftest import FROZEN_PARAMETERS

pytestmark = [pytest.mark.gui]

REGIONS = "-12:-8; -3:-2; 1:2"


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _capture_minimiser(monkeypatch):
    """Replace minimi_hi by a 'fit' that returns the start vector; returns the
    list its calls are recorded in."""
    calls = []

    def fake(func, x, y, p0, **kwargs):
        calls.append({'x': np.array(x, dtype=float), 'y': np.array(y, dtype=float),
                      'p0': np.array(p0, dtype=float), 'model': np.array(func(x, p0), dtype=float),
                      **kwargs})
        p0 = np.array(p0, dtype=float)
        fixed = {int(i) for i in np.ravel(kwargs['fix'])}
        free = [i for i in range(len(p0)) if i not in fixed]
        errors = np.full(len(p0), np.nan)
        errors[free] = 0.01
        return p0, errors, 1.0, np.eye(len(free)) * 1e-4

    monkeypatch.setattr(fitting_io.mi, 'minimi_hi', fake)
    return calls


def _spectra(app, tmp_path, n=1):
    """*n* copies of the frozen calibration spectrum as .dat files (6 decimals,
    as the program writes spectra), written into the path box."""
    A, B = np.loadtxt(os.path.join(FROZEN_PARAMETERS, "Calibration.dat"), comments='#', unpack=True)
    paths = []
    for k in range(n):
        path = str(tmp_path / f"spectrum_{k + 1}.dat")
        save_spectrum_with_metadata(path, A, B, [])
        paths.append(path)
    app.process_path.setPlainText(repr(paths))
    app.initialize_parameters()
    return paths


def _apply(app, text):
    """Switch "Apply exclusion" on and apply *text*."""
    if not app.toolbar.exclusion_action.isChecked():
        app.toolbar.exclusion_action.trigger()
    app.exclusion_bar.text.setText(text)
    assert app.exclusion_bar.commit()
    return app.exclusion_regions_in_use()


def _loaded(app, path):
    A, B = load_spectrum(app, [path], calibration_path=app.calibration_path)
    return np.asarray(A[0], dtype=float), np.asarray(B[0], dtype=float)


def test_the_minimiser_gets_only_the_points_outside_the_regions(physics_app, tmp_path, monkeypatch):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    path, = _spectra(app, tmp_path)
    regions = _apply(app, REGIONS)
    A, B = _loaded(app, path)
    keep = er.kept(A, regions)
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, path, _SerialPool())

    assert result['success'], result.get('message')
    assert 0 < keep.sum() < len(A)
    assert np.array_equal(calls[0]['x'], A[keep]) and np.array_equal(calls[0]['y'], B[keep])
    # the curves are drawn on every point
    assert len(result['A']) == len(result['SPC_f']) == len(A)
    assert result['exclusion_regions'] == regions
    n_free = len(calls[0]['p0']) - len(np.ravel(calls[0]['fix']))
    assert result['chi2_spread'] == fitting_io.chi2_spread(int(keep.sum()), n_free)


def test_the_model_at_the_fitted_points_is_the_full_model_there(physics_app, tmp_path, monkeypatch):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    path, = _spectra(app, tmp_path)
    regions = _apply(app, REGIONS)
    A, _ = _loaded(app, path)
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, path, _SerialPool())

    assert np.array_equal(calls[0]['model'], np.asarray(result['SPC_f'], dtype=float)[er.kept(A, regions)])


def test_without_regions_every_point_is_fitted(physics_app, tmp_path, monkeypatch):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    path, = _spectra(app, tmp_path)
    app.exclusion_bar.text.setText(REGIONS)
    app.exclusion_bar.commit()                 # typed, but "Apply exclusion" is off
    A, B = _loaded(app, path)
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, path, _SerialPool())

    assert app.exclusion_regions_in_use() == ()
    assert np.array_equal(calls[0]['x'], A) and np.array_equal(calls[0]['y'], B)
    assert result['exclusion_regions'] == ()


def test_the_regions_fit_like_the_spectrum_saved_without_their_points(physics_app, tmp_path, monkeypatch):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    path, = _spectra(app, tmp_path)
    regions = _apply(app, REGIONS)
    calls = _capture_minimiser(monkeypatch)
    fitting_io.fit_single_spectrum(app, path, _SerialPool())

    saved = str(tmp_path / "excl_spectrum_1.dat")
    assert process_without_excluded(app, path, saved, regions) == len(calls[0]['x'])
    app.toolbar.exclusion_action.trigger()      # off: the saved file has no excluded points left
    fitting_io.fit_single_spectrum(app, saved, _SerialPool())

    assert np.array_equal(calls[1]['x'], calls[0]['x'])
    assert np.array_equal(calls[1]['y'], calls[0]['y'])
    assert np.array_equal(calls[1]['model'], calls[0]['model'])


def test_a_fit_with_more_free_parameters_than_points_left_is_not_started(physics_app, tmp_path, monkeypatch):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    path, = _spectra(app, tmp_path)
    A, _ = _loaded(app, path)
    v = np.sort(A)
    # everything but the three fastest points
    _apply(app, f"{v[0] - 1}:{(v[-4] + v[-3]) / 2}")
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, path, _SerialPool())

    assert not result['success'] and not calls
    assert result['message'].startswith("3 points are left after the exclusion regions, but ")
    assert "the fit was not started" in result['message']


def test_a_spectrum_with_every_point_excluded_is_named(physics_app, tmp_path, monkeypatch):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    path, = _spectra(app, tmp_path)
    _apply(app, "-100:100")
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, path, _SerialPool())

    assert not result['success'] and not calls
    assert "spectrum_1.dat" in result['message']


def test_the_free_parameters_counted_are_those_the_fit_varies(physics_app, tmp_path, monkeypatch):
    app = physics_app
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Doublet')
    path, = _spectra(app, tmp_path)
    calls = _capture_minimiser(monkeypatch)
    fitting_io.fit_single_spectrum(app, path, _SerialPool())
    inputs = fitting_io.read_fit_inputs(app)

    assert fitting_io.free_parameter_count(inputs) == len(calls[0]['p0']) - len(np.ravel(calls[0]['fix']))


def test_nbaseline_spectra_each_keep_their_own_points(physics_app, tmp_path, monkeypatch):
    app = physics_app
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Doublet')
    paths = _spectra(app, tmp_path, 2)
    regions = _apply(app, REGIONS)
    A, B = _loaded(app, paths[0])
    keep = er.kept(A, regions)
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, paths[0], _SerialPool())

    assert result['success'], result.get('message')
    assert np.array_equal(calls[0]['x'], np.concatenate([A[keep], A[keep]]))
    assert np.array_equal(calls[0]['y'], np.concatenate([B[keep], B[keep]]))
    assert all(len(a) == len(A) for a in result['A_list'])
    assert result['exclusion_regions'] == regions


def test_a_recon_adds_its_smoothness_rows_after_the_fitted_points(physics_app, tmp_path, monkeypatch):
    from test_recon_mdl import _build_sextet_recon
    app = physics_app
    _build_sextet_recon(app, 5, "0.1,0.3,0.4,0.15,0.05")
    path, = _spectra(app, tmp_path)
    regions = _apply(app, REGIONS)
    A, B = _loaded(app, path)
    keep = er.kept(A, regions)
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, path, _SerialPool())

    assert result['success'], result.get('message')
    n_reg = calls[0]['n_reg']
    assert n_reg > 0
    assert len(calls[0]['x']) == keep.sum() + n_reg
    assert np.array_equal(calls[0]['y'][:keep.sum()], B[keep])


def test_a_one_model_fit_uses_the_regions(physics_app, tmp_path, monkeypatch):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    paths = _spectra(app, tmp_path, 2)
    regions = _apply(app, REGIONS)
    A, B = _loaded(app, paths[0])
    keep = er.kept(A, regions)
    calls = _capture_minimiser(monkeypatch)

    result = one_model.fit(app, one_model.read_template(app),
                           one_model.spectra_of([(p, ()) for p in paths]), _SerialPool())

    assert result['success'], result.get('message')
    assert np.array_equal(calls[0]['x'], np.concatenate([A[keep], A[keep]]))
    assert result['one_model'].exclusion_regions_of(1) == regions


def test_a_sequence_whose_spectrum_keeps_too_few_points_is_not_started(physics_app, tmp_path):
    app = physics_app
    app.params_table.select_model(1, 'Sextet')
    paths = _spectra(app, tmp_path, 2)
    A, _ = _loaded(app, paths[0])
    v = np.sort(A)
    _apply(app, f"{v[0] - 1}:{(v[-4] + v[-3]) / 2}")
    app.save_path.setText(str(tmp_path / "run"))
    app.inprogress = True

    app.start_sequential_fitting(paths, [(p, ()) for p in paths])

    assert not app.inprogress
    assert getattr(app, 'sequential_fitting_thread', None) is None
    status = app.log.toPlainText()
    assert status.startswith("Sequence fitting was not started")
    assert "spectrum_1.dat: 3 points are left" in status and "spectrum_2.dat" in status
