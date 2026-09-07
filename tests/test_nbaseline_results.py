"""Result-file output for a simultaneous (Nbaseline) fit.

Saving used to crash here: ``_save_graf_file`` received the per-spectrum LISTS
that the simultaneous plot path stores in ``last_fitting_data`` and fed them
straight into the scalar baseline formula (``TypeError: can't multiply sequence
by non-int``). The graf file must instead carry a full ``S<n>_`` block per
spectrum — velocity, data, that spectrum's OWN baseline, the fit and every
subspectrum — so the user can plot all of it later.
"""
import os

import numpy as np
import pytest

from syncmoss.spectrum_plotter import calculate_baseline
from syncmoss.constants import number_of_baseline_parameters as NB


def _read_graf(path):
    """Return (header list, 2-D data array) of a saved graf file."""
    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
    data = np.genfromtxt(path, delimiter='\t', skip_header=1)
    return header, np.atleast_2d(data)


def _simultaneous_fixture(app):
    """Two spectra of DIFFERENT length: Sextet + Nbaseline + Doublet."""
    model_list = ['baseline', 'Sextet', 'Nbaseline', 'Doublet']
    # flat p: baseline(8) Sextet(14) | Nbaseline(8) Doublet(9)
    p = np.arange(NB + 14 + NB + 9, dtype=float) + 1.0
    begining_spc = [0, NB + 14]

    A1 = np.linspace(-5.0, 5.0, 12)
    A2 = np.linspace(-3.0, 3.0, 17)
    B1, B2 = np.full_like(A1, 1000.0), np.full_like(A2, 900.0)
    SPC1, SPC2 = np.full_like(A1, 990.0), np.full_like(A2, 890.0)
    FS1, FS2 = [np.full_like(A1, 10.0)], [np.full_like(A2, 20.0)]

    app.last_fitting_data = {
        'A': [A1, A2], 'B': [B1, B2], 'SPC_f': [SPC1, SPC2], 'FS': [FS1, FS2],
        'is_simultaneous': True,
        'begining_spc': begining_spc,
        'spectrum_files': ['/data/first.dat', '/data/second.dat'],
    }
    app.last_plot_data = {
        'model': model_list, 'p': p,
        'Distri': [], 'Cor': [], 'Recon': [],
        'model_colors': ['blue', 'red', 'green', 'orange'],
        'is_simultaneous': True,
    }
    app.results_table.current_parameters = p
    app.results_table.current_model_list = model_list
    return p, (A1, A2)


@pytest.mark.gui
def test_graf_file_has_a_block_per_spectrum(physics_app, tmp_path):
    app = physics_app
    p, (A1, A2) = _simultaneous_fixture(app)

    path = os.path.join(str(tmp_path), 'sim_graf.txt')
    app._save_graf_file(path, app.last_fitting_data)
    header, data = _read_graf(path)

    assert header == [
        'S1_Velocity', 'S1_Data', 'S1_Baseline', 'S1_Fit', 'S1_Sextet',
        'S2_Velocity', 'S2_Data', 'S2_Baseline', 'S2_Fit', 'S2_Doublet',
    ]

    # Rows are padded to the longest spectrum; the short one ends in NaN.
    assert data.shape[0] == len(A2)
    col = {name: data[:, i] for i, name in enumerate(header)}
    assert np.allclose(col['S1_Velocity'][:len(A1)], A1)
    assert np.all(np.isnan(col['S1_Velocity'][len(A1):]))
    assert np.allclose(col['S2_Velocity'], A2)
    assert np.allclose(col['S1_Sextet'][:len(A1)], 10.0)
    assert np.allclose(col['S2_Doublet'], 20.0)


@pytest.mark.gui
def test_graf_baselines_use_each_section_own_parameters(physics_app, tmp_path):
    """Spectrum 2's baseline comes from its Nbaseline slots, not from p[0:8]."""
    app = physics_app
    p, (A1, A2) = _simultaneous_fixture(app)

    path = os.path.join(str(tmp_path), 'sim_graf.txt')
    app._save_graf_file(path, app.last_fitting_data)
    header, data = _read_graf(path)
    col = {name: data[:, i] for i, name in enumerate(header)}

    expected1 = calculate_baseline(p[0:], A1)
    expected2 = calculate_baseline(p[NB + 14:], A2)
    assert np.allclose(col['S1_Baseline'][:len(A1)], expected1, rtol=1e-6)
    assert np.allclose(col['S2_Baseline'], expected2, rtol=1e-6)
    # The two baselines really are different (guards against silently reusing p[0:8]).
    assert not np.allclose(expected2, calculate_baseline(p[0:], A2))


@pytest.mark.gui
def test_single_spectrum_graf_header_unchanged(physics_app, tmp_path):
    """The single-spectrum block keeps its historical unprefixed column names."""
    app = physics_app
    A = np.linspace(-5.0, 5.0, 10)
    p = np.arange(NB + 14, dtype=float) + 1.0
    app.last_fitting_data = {
        'A': A, 'B': np.full_like(A, 1000.0), 'SPC_f': np.full_like(A, 990.0),
        'FS': [np.full_like(A, 10.0)], 'is_simultaneous': False,
    }
    app.last_plot_data = {'model': ['Sextet'], 'p': p, 'Distri': [], 'Cor': [], 'Recon': []}
    app.results_table.current_parameters = p
    app.results_table.current_model_list = ['baseline', 'Sextet']

    path = os.path.join(str(tmp_path), 'one_graf.txt')
    app._save_graf_file(path, app.last_fitting_data)
    header, data = _read_graf(path)

    assert header == ['Velocity', 'Data', 'Baseline', 'Fit', 'Sextet']
    assert np.allclose(data[:, 2], calculate_baseline(p, A), rtol=1e-6)


@pytest.mark.gui
def test_param_file_names_every_simultaneous_spectrum(physics_app, tmp_path):
    """The single result row of a simultaneous fit lists all its spectra."""
    app = physics_app
    _simultaneous_fixture(app)
    app.results_table.current_errors = np.zeros_like(app.results_table.current_parameters)
    app.results_table.current_parameter_names = [
        ['Ns', 'Os', 'c2s', 'lins', 'Nnr', 'Onr', 'c2nr', 'linnr'],
    ]
    app.results_table.current_chi2 = 1.0
    app.save_path.setText(os.path.join(str(tmp_path), 'sim'))
    app._save_result_files('new')

    with open(os.path.join(str(tmp_path), 'sim_param.txt'), encoding='utf-8') as f:
        f.readline()
        row = f.readline().split('\t')
    assert row[0] == 'first.dat; second.dat'
