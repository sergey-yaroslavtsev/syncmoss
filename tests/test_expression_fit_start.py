"""A fit starts from the model Show model draws, Expressions included.

``read_model`` leaves 0 in an Expression's slot, and ``minimi_hi`` computes its
first model and Jacobian from the start vector exactly as it is given. A
parameter linked to an Expression (``=[k,1]``) therefore used to start at 0,
whatever the Expression said. When the model with that parameter at 0 fitted
the data at least as well as the Expression's value, every step was compared
against a start no consistent point can reach: the fit never moved and handed
the parameter back as 0 -- the forced value silently lost.

The spectrum is the bundled alpha-Fe calibration spectrum (a temp copy, see
conftest), whose delta is near 0, and the Expression forces delta = 0.3. That
the start vector carries the evaluated Expression and its links is pinned in
test_fit_core.test_a_single_fit_hands_the_table_to_the_minimiser; here a real
fit must keep the forced value.
"""
import pytest

from syncmoss import fitting_io, model_io
from syncmoss.constants import number_of_baseline_parameters as NB

from conftest import redirect_params_dir_to_tmp, SerialPool

pytestmark = [pytest.mark.gui]

DELTA = NB + 1      # flat index of the Sextet's delta (row 1, column 1)


def _forced_delta_model(app, tmp_path, text='0.3'):
    """Sextet + Expression row, the Sextet's delta linked to the Expression.

    Returns the flat index of the Expression's slot.
    """
    redirect_params_dir_to_tmp(app, tmp_path)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, text, '', '', 'False')
    slot = int(model_io.read_model(app)[8][0])
    model_io._set_row_param(pt.row_widgets[1], 1, f'=[{slot},1]', '', '', 'False')
    app.initialize_parameters()          # what fit_pressed does first
    return slot


def test_a_forced_value_survives_the_fit(physics_app, tmp_path):
    """Only the baseline Ns is free: before the fix this fit froze at delta = 0."""
    app = physics_app
    slot = _forced_delta_model(app, tmp_path)
    row = app.params_table.row_widgets[1]
    for col in range(14):
        if col == 1:
            continue                     # the link
        param_widget = row.layout().itemAt(col + 1).widget()
        param_widget.layout().itemAt(0).layout().itemAt(1).widget().setChecked(True)
    start_ns = model_io.read_model(app)[1][0]

    result = fitting_io.fit_single_spectrum(app, app.calibration_path, SerialPool())

    assert result['success'], result.get('message')
    p = result['parameters']
    assert p[slot] == pytest.approx(0.3)
    assert p[DELTA] == pytest.approx(0.3)
    assert p[0] != start_ns              # the free parameter moved
