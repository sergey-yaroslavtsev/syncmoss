"""Result-file output for Recon / Distr / Corr distributions.

Covers the three additions to the saved artifacts:
  * ``distribution_curves`` — the extra graf.txt columns (x/y per distribution and
    correlation, in model order with per-type indices),
  * ``_save_parameters_file`` — the Recon 'weights' column is written as a
    placeholder ``1`` (it is not a scalar parameter),
  * ``_save_graf_file`` / ``_save_distributions_png`` — the graf file gains the
    distribution columns and a ``*_distributions.png`` is produced.
"""
import os

import numpy as np
import pytest

from syncmoss.spectrum_plotter import distribution_curves
from syncmoss.constants import number_of_baseline_parameters as NB


def test_distribution_curves_column_order():
    """model = Sextet, Distr, Corr, Distr, Recon, Recon -> exactly the columns
    Distr_x_1 Distr_y_1 Corr_x_1 Corr_y_1 Distr_x_2 Distr_y_2 Recon_x_1 Recon_y_1
    Recon_x_2 Recon_y_2 (per-type counters, model order; the Corr binds to Distr 1)."""
    model = ['Sextet', 'Distr', 'Corr', 'Distr', 'Recon', 'Recon']
    # p offsets: baseline(8) + Sextet(14) then Distr(5) Corr(2) Distr(5) Recon(7) Recon(7)
    p = np.zeros(NB + 14 + 5 + 2 + 5 + 7 + 7)
    d1 = NB + 14              # Distr 1 slots: par,L,R,Num,ph
    p[d1 + 1], p[d1 + 2], p[d1 + 3] = 0.0, 1.0, 4      # L,R,Num
    c1 = d1 + 5               # Corr slots: par,ph
    p[c1] = 1.0
    d2 = c1 + 2               # Distr 2
    p[d2 + 1], p[d2 + 2], p[d2 + 3] = 0.0, 2.0, 5
    r1 = d2 + 5               # Recon 1 slots: par,L,R,Num,D_dif,D_dif2,ph
    p[r1 + 1], p[r1 + 2], p[r1 + 3] = 0.0, 1.0, 3
    r2 = r1 + 7               # Recon 2
    p[r2 + 1], p[r2 + 2], p[r2 + 3] = 0.0, 3.0, 6

    Distri = ['1', '1']       # flat PDFs -> uniform density
    Cor = ['X']               # identity dependency
    Recon = [np.ones(3), np.ones(6)]

    cols = distribution_curves(model, p, Distri, Cor, Recon)
    names = [name for name, _ in cols]
    assert names == [
        'Distr_x_1', 'Distr_y_1', 'Corr_x_1', 'Corr_y_1',
        'Distr_x_2', 'Distr_y_2', 'Recon_x_1', 'Recon_y_1',
        'Recon_x_2', 'Recon_y_2',
    ]

    values = dict(cols)
    assert np.allclose(values['Distr_x_1'], np.linspace(0, 1, 4))
    assert np.allclose(values['Distr_y_1'], 0.25)           # normalized flat PDF
    assert np.allclose(values['Corr_x_1'], np.linspace(0, 1, 4))   # eval('X') over the grid
    assert np.allclose(values['Corr_y_1'], 0.25)            # parent density
    assert np.allclose(values['Distr_x_2'], np.linspace(0, 2, 5))
    assert np.allclose(values['Distr_y_2'], 0.2)
    assert np.allclose(values['Recon_x_1'], np.linspace(0, 1, 3))
    assert np.allclose(values['Recon_y_1'], 1.0 / 3.0)      # normalized weights
    assert np.allclose(values['Recon_x_2'], np.linspace(0, 3, 6))
    assert np.allclose(values['Recon_y_2'], 1.0 / 6.0)


def test_distribution_curves_empty_without_distributions():
    assert distribution_curves(['Sextet'], np.zeros(NB + 14), [], [], []) == []


def test_param_file_writes_placeholder_for_recon_weights(physics_app, tmp_path):
    """The Recon 'weights' column is written as 1 (not the placeholder p-slot)."""
    model_list = ['baseline', 'Recon']
    parameter_names = [
        ['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr'],
        ['par', 'L', 'R', 'Num', 'D_dif', 'D_dif2', 'weights'],
    ]
    parameters = np.arange(NB + 7, dtype=float)   # weights slot would be 14.0
    errors = np.zeros(NB + 7, dtype=float)

    path = os.path.join(str(tmp_path), 'r_param.txt')
    physics_app._save_parameters_file(path, parameters, errors, model_list,
                                      parameter_names, 'spec.dat', 1.23, 'new')

    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        data = f.readline().rstrip('\n').split('\t')
    iw = header.index('weights')
    assert data[iw] == '1'                 # placeholder, not '14.0'
    assert data[header.index('d_weights')] == 'nan'


def test_graf_and_distributions_png_for_recon(physics_app, tmp_path):
    """A single-spectrum result with a Recon adds Recon_x/Recon_y graf columns and
    writes a *_distributions.png."""
    app = physics_app
    A = np.linspace(-5.0, 5.0, 24)
    B = np.full_like(A, 1000.0)
    SPC = np.full_like(A, 1000.0)
    FS = [np.full_like(A, 10.0)]
    app.last_fitting_data = {'A': A, 'B': B, 'SPC_f': SPC, 'FS': FS}

    model = ['Sextet', 'Recon']
    p = np.zeros(NB + 14 + 7)
    r = NB + 14
    p[r + 1], p[r + 2], p[r + 3] = 0.0, 1.0, 5           # L, R, Num
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    app.last_plot_data = {
        'model': model, 'p': p, 'p_flat': p,
        'Distri': [], 'Cor': [], 'Recon': [weights],
        'model_colors': ['blue', 'red'],
    }
    app.results_table.current_parameters = p
    app.results_table.current_model_list = model

    graf = os.path.join(str(tmp_path), 'g_graf.txt')
    app._save_graf_file(graf, app.last_fitting_data)
    with open(graf, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
    assert header[:4] == ['Velocity', 'Data', 'Baseline', 'Fit']
    assert 'Recon_x_1' in header and 'Recon_y_1' in header

    png = os.path.join(str(tmp_path), 'g_distributions.png')
    app._save_distributions_png(png)
    assert os.path.exists(png) and os.path.getsize(png) > 0
