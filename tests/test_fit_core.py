"""What fit_single_spectrum gives the minimiser, and what it hands back.

Pinned before the core of the fit was split off (fitting_io.fit_model) so the
simultaneous one-model fit could feed it a model that is not in the table: the
start vector, fixes, links, bounds and expressions the table makes, and the
per-spectrum pieces of the result, for a single and for an Nbaseline fit.

The minimiser is replaced by one that returns the start vector, so no time is
spent fitting; the curves after it are computed for real (serial pool).
"""
import os
import shutil

import numpy as np
import pytest

from syncmoss import fitting_io, model_io
from syncmoss.constants import number_of_baseline_parameters as NB

from conftest import redirect_params_dir_to_tmp

pytestmark = [pytest.mark.gui]

SEXTET, DOUBLET = 14, 9


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
                      'p0': np.array(p0, dtype=float), **kwargs})
        p0 = np.array(p0, dtype=float)
        fixed = {int(i) for i in np.ravel(kwargs['fix'])}
        free = [i for i in range(len(p0)) if i not in fixed]
        errors = np.full(len(p0), np.nan)
        errors[free] = 0.01
        return p0, errors, 1.0, np.eye(len(free)) * 1e-4

    monkeypatch.setattr(fitting_io.mi, 'minimi_hi', fake)
    return calls


def _expected_start(app):
    """The fit's inputs as the table defines them (links and expressions applied)."""
    model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = \
        model_io.read_model(app)
    bounds, fix = model_io.read_bounds_and_fix(app, len(p))
    fix = np.unique(np.concatenate([fix, con1, DistriN, NExpr, ReconN]).astype(int))
    p0 = np.array(p, dtype=float)
    for i in range(len(con1)):
        p0[int(con1[i])] = p0[int(con2[i])] * con3[i]
    for e in range(len(Expr)):
        p0[NExpr[e]] = eval(Expr[e], {'p': p0})
        for c in np.where(con2 == NExpr[e])[0]:
            p0[int(con1[c])] = p0[int(con2[c])] * con3[c]
    return {'model': model, 'p0': p0, 'fix': fix, 'bounds': bounds,
            'confu': np.array([con1, con2, con3]), 'Expr': Expr, 'NExpr': NExpr}


def _spectrum_copies(app, tmp_path, n):
    """*n* copies of the bundled spectrum, written into the path box."""
    paths = []
    for k in range(n):
        path = str(tmp_path / f"spectrum_{k + 1}.dat")
        shutil.copy2(app.calibration_path, path)
        paths.append(path)
    app.process_path.setPlainText(repr(paths))
    return paths


def test_a_single_fit_hands_the_table_to_the_minimiser(physics_app, tmp_path, monkeypatch):
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, 'p[9]*2', '', '', 'False')
    slot = NB + SEXTET                                   # the Expression's slot
    model_io._set_row_param(pt.row_widgets[1], 2, f'=[{slot},1]', '', '', 'False')
    model_io._set_row_param(pt.row_widgets[1], 1, '0.05', '-1', '1', 'False')
    app.initialize_parameters()
    expected = _expected_start(app)
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, app.calibration_path, _SerialPool())

    assert result['success'], result.get('message')
    call = calls[0]
    assert np.array_equal(call['p0'], expected['p0'])
    assert call['p0'][slot] == pytest.approx(0.1) and call['p0'][NB + 2] == pytest.approx(0.1)
    assert np.array_equal(np.sort(np.ravel(call['fix'])), expected['fix'])
    assert np.array_equal(call['confu'], expected['confu'])
    assert np.array_equal(call['bounds'], expected['bounds'])
    assert list(call['Expr']) == ['p[9]*2'] and list(call['NExpr']) == [slot]
    assert call['n_reg'] == 0
    assert len(call['x']) == len(call['y']) == len(result['A'])
    assert result['is_simultaneous'] is False
    assert len(result['FS']) == 1 and len(result['SPC_f']) == len(result['A'])
    assert np.array_equal(result['parameters'], expected['p0'])


# --- TI cuts the joined spectra at their lengths -----------------------------------
# TI used to find where one spectrum ends and the next begins from the velocity
# steps (a new spectrum wherever a step has not the sign of the first one); a
# spectrum running the other way was cut into single points. It is now told the
# lengths. The bundled spectrum, its instrumental function and an Nbaseline model
# are used as they are; JN is small.

JN = 16


def _ti_inputs(app, tmp_path):
    """(the spectrum's velocities, TI's arguments) for Sextet | Nbaseline + Doublet."""
    from syncmoss.instrumental_io import resolve_instrumental_for_file, compute_norm
    paths = _spectrum_copies(app, tmp_path, 2)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Doublet')
    model, p = model_io.read_model(app)[:2]
    pool = _SerialPool()
    method = resolve_instrumental_for_file(app, paths[0], use_dat_metadata=True)
    method['Norm'] = compute_norm(pool, JN, method)
    A = fitting_io.load_spectrum(app, paths[0], calibration_path=app.calibration_path)[0][0]
    return A, model, p, pool, method


def _ti(x, model, p, pool, method, **kwargs):
    from syncmoss import models
    return models.TI(np.asarray(x), p, model, JN, pool, method['x0'], method['MulCo'],
                     method['INS'], [0], [0], Met=method['Met'], Norm=method['Norm'], **kwargs)


def test_cut_at_the_lengths_ti_gives_the_same_for_spectra_in_one_direction(physics_app, tmp_path):
    A, model, p, pool, method = _ti_inputs(physics_app, tmp_path)
    x = np.concatenate([A, A[:300]])                    # the second one shorter
    guessed = _ti(x, model, p, pool, method)
    told = _ti(x, model, p, pool, method, lengths=[len(A), 300])
    assert np.array_equal(guessed, told)                # bit for bit


def test_ti_takes_a_spectrum_running_the_other_way(physics_app, tmp_path):
    A, model, p, pool, method = _ti_inputs(physics_app, tmp_path)
    n = len(A)
    forward = _ti(np.concatenate([A, A]), model, p, pool, method, lengths=[n, n])
    backward = _ti(np.concatenate([A, A[::-1]]), model, p, pool, method, lengths=[n, n])
    assert np.array_equal(backward[:n], forward[:n])
    assert np.array_equal(backward[n:], forward[n:][::-1])   # the same points, other order


def test_per_spectrum_instrumental_functions_take_the_lengths_too(physics_app, tmp_path):
    """The list form of the instrumental arguments (a CMS + SMS mix) cuts the
    same way; with the same function twice it is the plain call."""
    A, model, p, pool, method = _ti_inputs(physics_app, tmp_path)
    from syncmoss import models
    n = len(A)
    x = np.concatenate([A, A[::-1]])
    plain = _ti(x, model, p, pool, method, lengths=[n, n])
    listed = models.TI(x, p, model, JN, pool, [method['x0']] * 2, [method['MulCo']] * 2,
                       [method['INS']] * 2, [0], [0], Met=[method['Met']] * 2,
                       Norm=[method['Norm']] * 2, lengths=[n, n])
    assert np.array_equal(plain, listed)


def test_lengths_that_do_not_fit_are_refused(physics_app, tmp_path):
    A, model, p, pool, method = _ti_inputs(physics_app, tmp_path)
    with pytest.raises(ValueError):
        _ti(np.concatenate([A, A]), model, p, pool, method, lengths=[len(A), len(A) - 1])


def test_a_spectrum_running_the_other_way_fits_like_one_running_forward(physics_app, tmp_path,
                                                                       monkeypatch):
    """The same two spectra, the second stored backwards: the same fit.

    Only Ns, T and delta are free, starting near the answer. With everything
    free the fit is so ill-conditioned that the other order of the same points
    in the sums already moves its poorly determined parameters by more than
    1e-6; and from Ns = 10000 one spectrum's delta runs away (to ~-1e8) --
    which one depends on the order of the points, with the old cutting as well.
    """
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    paths = _spectrum_copies(app, tmp_path, 2)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Sextet')
    for row in (1, 3):
        for col in range(2, SEXTET):
            fix_box = pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(0).layout().itemAt(1).widget()
            fix_box.setChecked(True)
        model_io._set_row_param(pt.row_widgets[row], 0, '6.5', '0', '', 'False')    # T
    for row in (0, 2):
        model_io._set_row_param(pt.row_widgets[row], 0, '6000', '1', '', 'False')   # Ns
    app.jn0_input.setText(str(JN))
    app.initialize_parameters()
    start = model_io.read_model(app)[1]

    forward = fitting_io.fit_single_spectrum(app, paths[0], _SerialPool())
    real_load = fitting_io.load_spectrum

    def second_reversed(app_, path, **kwargs):
        A, B = real_load(app_, path, **kwargs)
        return ([a[::-1] for a in A], [b[::-1] for b in B]) if path == paths[1] else (A, B)

    monkeypatch.setattr(fitting_io, 'load_spectrum', second_reversed)
    backward = fitting_io.fit_single_spectrum(app, paths[0], _SerialPool())

    assert forward['success'] and backward['success'], backward.get('message')
    assert np.allclose(backward['parameters'], forward['parameters'], rtol=1e-6, atol=1e-9)
    assert backward['chi2'] == pytest.approx(forward['chi2'], rel=1e-6)
    curve_back = np.asarray(backward['SPC_f_list'][1], dtype=float)
    curve_forward = np.asarray(forward['SPC_f_list'][1], dtype=float)
    assert np.allclose(curve_back, curve_forward[::-1], rtol=1e-6)
    assert forward['parameters'][0] != start[0]          # the fit did move


def test_show_model_draws_a_spectrum_running_the_other_way_as_one(physics_app, tmp_path,
                                                                  monkeypatch):
    """Its plot cuts the joined spectra at their lengths too: two panels."""
    import syncmoss.syncmoss_main as sm
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    paths = _spectrum_copies(app, tmp_path, 2)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Sextet')
    app.jn0_input.setText(str(JN))
    app.pool = _SerialPool()
    backwards = os.path.abspath(paths[1])
    real_load = sm.load_spectrum

    def second_reversed(app_, files, **kwargs):
        A, B = real_load(app_, files, **kwargs)
        first = files[0] if isinstance(files, list) else files
        if os.path.abspath(first) == backwards:
            return [a[::-1] for a in A], [b[::-1] for b in B]
        return A, B

    monkeypatch.setattr(sm, 'load_spectrum', second_reversed)
    monkeypatch.setattr(sm.ShowModelThread, 'start', lambda thread: thread.run())
    app.showM_pressed()
    assert "red" not in app.log.styleSheet().lower(), app.log.toPlainText()
    assert app.last_plot_data['lengths'] == [len(app.last_plot_data['A']) // 2] * 2
    assert len(app.figure.axes) == 2


def test_an_nbaseline_fit_hands_every_section(physics_app, tmp_path, monkeypatch):
    app = physics_app
    redirect_params_dir_to_tmp(app, tmp_path)
    paths = _spectrum_copies(app, tmp_path, 2)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Doublet')
    app.initialize_parameters()
    expected = _expected_start(app)
    calls = _capture_minimiser(monkeypatch)

    result = fitting_io.fit_single_spectrum(app, paths[0], _SerialPool())

    assert result['success'], result.get('message')
    call = calls[0]
    assert np.array_equal(call['p0'], expected['p0'])
    assert len(call['p0']) == NB + SEXTET + NB + DOUBLET
    assert np.array_equal(np.sort(np.ravel(call['fix'])), expected['fix'])
    assert np.array_equal(call['bounds'], expected['bounds'])
    lengths = [len(a) for a in result['A_list']]
    assert len(call['x']) == sum(lengths) and lengths[0] == lengths[1] > 0
    assert result['is_simultaneous'] is True
    assert result['begining_spc'] == [0, NB + SEXTET]
    assert result['model_separate'] == [['Sextet'], ['Doublet']]
    assert [len(fs) for fs in result['FS_list']] == [1, 1]
    assert [len(s) for s in result['SPC_f_list']] == lengths
    assert result['spectrum_files'] == paths
