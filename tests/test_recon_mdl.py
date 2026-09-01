"""Recon .mdl round-trip and read_model layout (offscreen Qt via physics_app).

The reconstruction weights ride in a single fixed placeholder slot of the flat p
(so parameter counting / links / p[i] expressions never shift with Num); their
values live in the parallel ``Recon`` list, serialized as a comma-separated
string in the trailing weight column. These pin that read_model parses them and
that a save -> load cycle preserves them (the fixed 7-slot footprint means the
generic .mdl loader needs no Num-dependent special-casing).
"""
import numpy as np
import pytest

from syncmoss.model_io import read_model, _save_model_to_file, load_model_from_path
from syncmoss.constants import number_of_baseline_parameters

# Every test here builds the PySide6 GUI via the physics_app fixture.
pytestmark = pytest.mark.gui


def _model_btn(pt, row):
    return pt.row_widgets[row].layout().itemAt(0).widget().layout().itemAt(1).widget()


def _model_name(pt, row):
    return _model_btn(pt, row).text()


def _set_value(pt, row, col, text):
    """Set the value QLineEdit of parameter column ``col`` (0-based)."""
    pw = pt.row_widgets[row].layout().itemAt(col + 1).widget()
    pw.layout().itemAt(1).widget().setText(text)


def _build_sextet_recon(physics_app, num, weights_text):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model_by_button('Recon', _model_btn(pt, 2))
    assert _model_name(pt, 2) == 'Recon'
    _set_value(pt, 2, 3, str(num))          # Num (col 3)
    _set_value(pt, 2, 6, weights_text)      # weight vector (col 6, the trailing text column)
    return pt


def test_recon_read_model_layout(physics_app):
    _build_sextet_recon(physics_app, 5, '0.1,0.2,0.4,0.2,0.1')
    model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = read_model(physics_app)

    assert model == ['Sextet', 'Recon']
    # Fixed footprint: baseline(8) + Sextet(14) + Recon(7) = 29, independent of Num.
    assert len(p) == number_of_baseline_parameters + 14 + 7
    assert len(Recon) == 1
    assert np.allclose(Recon[0], [0.1, 0.2, 0.4, 0.2, 0.1])
    # The weight placeholder slot is recorded (force-fixed by the fit) and is the
    # last Recon slot.
    assert len(ReconN) == 1
    assert int(ReconN[0]) == len(p) - 1
    # No PDF/dependency expressions for a Recon.
    assert list(Distri) == [] and list(Cor) == []


def test_recon_empty_weights_default_uniform(physics_app):
    """An empty weight field falls back to a uniform vector of length Num."""
    _build_sextet_recon(physics_app, 8, '')
    *_head, Recon, ReconN = read_model(physics_app)
    assert len(Recon) == 1
    assert Recon[0].size == 8
    assert np.allclose(Recon[0], 1.0 / 8)


def test_recon_mdl_save_load_roundtrip(physics_app, tmp_path):
    _build_sextet_recon(physics_app, 6, '0.05,0.1,0.35,0.35,0.1,0.05')
    path = str(tmp_path / 'recon_model.mdl')
    _save_model_to_file(physics_app, path)

    # Reset the model rows, then load the saved file back.
    pt = physics_app.params_table
    for row in range(1, len(pt.row_widgets)):
        pt.clear_row_params(row)
    load_model_from_path(physics_app, path)

    model, p, *_mid, Recon, ReconN = read_model(physics_app)
    assert model == ['Sextet', 'Recon']
    assert len(Recon) == 1
    assert np.allclose(Recon[0], [0.05, 0.1, 0.35, 0.35, 0.1, 0.05])
