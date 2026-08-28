"""GUI-level tests for the polarized ("thick") models, the Layer marker, the
Distr/Corr placement guard, and the default-locked orientation angles.

Head-less (offscreen Qt) via the ``physics_app`` fixture.
"""
import pytest

from syncmoss.model_io import read_model


def _model_btn(pt, row):
    return pt.row_widgets[row].layout().itemAt(0).widget().layout().itemAt(1).widget()


def _model_name(pt, row):
    return _model_btn(pt, row).text()


def test_layer_selection_has_no_parameters(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Layer')
    assert _model_name(pt, 1) == 'Layer'
    assert pt.row_params[1] == 0
    model = read_model(physics_app)[0]
    assert 'Layer' in model
    # Layer contributes no parameters to the flat array.
    assert model.count('Layer') == 1


def test_thick_model_selection_param_count(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    assert _model_name(pt, 1) == 'Sextet'
    assert pt.row_params[1] == 14  # T,d,e,H,L,G,theta_k,phi_h,A,a+,a-,GH,I1/I3,Am


def test_distr_after_baseline_is_blocked(physics_app):
    pt = physics_app.params_table
    btn = _model_btn(pt, 1)             # row 1, previous row is the baseline
    pt.select_model_by_button('Distr', btn)
    assert _model_name(pt, 1) == 'None'  # not entered


def test_distr_after_layer_is_blocked(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Layer')
    btn = _model_btn(pt, 2)
    pt.select_model_by_button('Distr', btn)
    assert _model_name(pt, 2) == 'None'


def test_distr_after_expression_is_blocked(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Expression')
    btn = _model_btn(pt, 2)
    pt.select_model_by_button('Distr', btn)
    assert _model_name(pt, 2) == 'None'


def test_distr_after_component_is_allowed(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Doublet')
    btn = _model_btn(pt, 2)
    pt.select_model_by_button('Distr', btn)
    assert _model_name(pt, 2) == 'Distr'


def test_corr_requires_distr_or_corr(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Doublet')
    btn = _model_btn(pt, 2)
    pt.select_model_by_button('Corr', btn)
    assert _model_name(pt, 2) == 'None'   # Corr after a plain model is blocked
    # but Corr after a Distr is fine
    pt.select_model(1, 'Doublet')
    btn2 = _model_btn(pt, 2)
    pt.select_model_by_button('Distr', btn2)
    btn3 = _model_btn(pt, 3)
    pt.select_model_by_button('Corr', btn3)
    assert _model_name(pt, 3) == 'Corr'


def test_recon_after_component_is_allowed(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    btn = _model_btn(pt, 2)
    pt.select_model_by_button('Recon', btn)
    assert _model_name(pt, 2) == 'Recon'
    # Recon shows its 7 fixed control slots (par, L, R, Num, D_dif, D_dif2, weights).
    assert pt.row_params[2] == 7


def test_recon_after_baseline_is_blocked(physics_app):
    pt = physics_app.params_table
    btn = _model_btn(pt, 1)             # row 1, previous row is the baseline
    pt.select_model_by_button('Recon', btn)
    assert _model_name(pt, 1) == 'None'  # not entered


def test_corr_after_recon_is_allowed(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    btn2 = _model_btn(pt, 2)
    pt.select_model_by_button('Recon', btn2)
    assert _model_name(pt, 2) == 'Recon'
    btn3 = _model_btn(pt, 3)
    pt.select_model_by_button('Corr', btn3)
    assert _model_name(pt, 3) == 'Corr'   # Corr may correlate onto a Recon


def test_recon_weight_column_hidden(physics_app):
    """The reconstruction weight vector is managed internally (fitted + shown in
    the Distribution plot), so the row shows only the 6 controls: the trailing
    weights column exists (fixed flat slot, round-trips) but is hidden."""
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model_by_button('Recon', _model_btn(pt, 2))
    assert _model_name(pt, 2) == 'Recon'
    row_layout = pt.row_widgets[2].layout()
    # cols 0..5 (par, L, R, Num, D_dif, D_dif2) visible; col 6 (weights) hidden.
    for col in range(6):
        assert row_layout.itemAt(col + 1).widget().isHidden() is False, f"col {col} should be visible"
    assert row_layout.itemAt(7).widget().isHidden() is True  # weights column hidden


def test_recon_num_and_reg_locked_by_default(physics_app):
    """par/Num/D_dif/D_dif2 are structural and hard-locked; L/R stay free."""
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    btn = _model_btn(pt, 2)
    pt.select_model_by_button('Recon', btn)
    # cols: 0 par, 1 L, 2 R, 3 Num, 4 D_dif, 5 D_dif2, 6 weights
    for col in (0, 3, 4, 5):
        assert _fix_cb(pt, 2, col).isChecked(), f"Recon col {col} should be locked"
    assert _fix_cb(pt, 2, 1).isChecked() is False   # L free
    assert _fix_cb(pt, 2, 2).isChecked() is False   # R free


# Orientation/texture parameters each polarized model carries that are FIXED by
# default, as {param column: label}; the user unticks the box to refine them.
# (theta_k, phi_h replace the former scalar asymmetry and are followed by the
# uniaxial texture parameter A; 'Hamiltonian' keeps its crystal angles, the
# beam-rotation alpha_k and the three mosaic order parameters A, Am, Ah -- its
# reference-orientation angles are locked WITH them because they do nothing in
# the default random-powder setting (A = Am = Ah = 0). The Faraday-active
# models also carry a magnetic polar-order Am right after A, likewise locked.)
_THICK_LOCKED_ANGLES = {
    'Doublet':     {5: 'θk, °', 6: 'φh, °', 7: 'A'},
    'Sextet':      {6: 'θk, °', 7: 'φh, °', 8: 'A', 9: 'Am'},
    'MDGD':        {10: 'θk, °', 11: 'φh, °', 12: 'A', 13: 'Am'},
    'Relax_MS':    {5: 'θk, °', 6: 'φh, °', 7: 'A'},
    'Relax_2S':    {8: 'θk, °', 9: 'φh, °', 10: 'A', 11: 'Am'},
    'Hamiltonian': {9: 'θ, °', 10: 'φ, °', 11: 'αk, °', 12: 'A', 13: 'Am', 14: 'Ah'},
    'ASM':         {9: 'θk, °', 10: 'φh, °', 11: 'A', 14: 'ω, °'},
    'SCDW':        {6: 'θk, °', 7: 'φh, °', 8: 'A', 9: 'Am'},
}


def _param_top_layout(pt, row, col):
    param_widget = pt.row_widgets[row].layout().itemAt(col + 1).widget()
    return param_widget.layout().itemAt(0).layout()


def _fix_cb(pt, row, col):
    return _param_top_layout(pt, row, col).itemAt(1).widget()


def _param_name(pt, row, col):
    return _param_top_layout(pt, row, col).itemAt(0).widget().text()


@pytest.mark.parametrize("model", sorted(_THICK_LOCKED_ANGLES))
def test_thick_extra_angles_locked_by_default(physics_app, model):
    pt = physics_app.params_table
    pt.select_model(1, model)
    assert _model_name(pt, 1) == model
    for col, label in _THICK_LOCKED_ANGLES[model].items():
        assert _param_name(pt, 1, col) == label, f"{model}: unexpected label at col {col}"
        cb = _fix_cb(pt, 1, col)
        assert cb.isChecked(), f"{model}: angle '{label}' should be fixed by default"
        # Locked but unlockable: the checkbox stays enabled so the user can refine it.
        assert cb.isEnabled(), f"{model}: angle '{label}' must remain user-unlockable"


# --- opening a model file written before the Hamiltonian merge ---------------
_BASELINE_ROW = '\t'.join(['10000', '1', '', '', 'False'] + ['0', '', '', '', 'True'] * 7)


def _write_legacy_mdl(tmp_path, name, values):
    """A minimal one-component .mdl in the on-disk format (names, colors, rows)."""
    row = '\t'.join('\t'.join([str(v), '', '', '', 'False']) for v in values)
    path = tmp_path / (name + '.mdl')
    path.write_text('\t'.join(['baseline', name]) + '\n'
                    + '\t'.join(['red', 'red']) + '\n'
                    + _BASELINE_ROW + '\n' + row + '\n', encoding='utf-8')
    return str(path)


@pytest.mark.parametrize("name,values,tail", [
    # old powder Hamiltonian -> Hamiltonian at (A, Am, Ah) = (0, 0, 0)
    ('Hamilton_pc', [1.0, 0.1, 0.4, 33.0, 0.098, 0.12, 0.3, 20.0, 30.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # very old single-crystal Hamiltonian (no alpha_k) -> (1, 1, 1), + alpha_k = 0
    ('Hamilton_mc', [1.0, 0.1, 0.4, 33.0, 0.098, 0.12, 0.3, 20.0, 30.0, 40.0, 50.0],
     [0.0, 1.0, 1.0, 1.0]),
    # current single-crystal layout (with alpha_k) -> (1, 1, 1)
    ('Hamilton_mc', [1.0, 0.1, 0.4, 33.0, 0.098, 0.12, 0.3, 20.0, 30.0, 40.0, 50.0, 70.0],
     [1.0, 1.0, 1.0]),
])
def test_loading_an_old_hamiltonian_model_file_upgrades_it(physics_app, tmp_path, name, values, tail):
    """Both old Hamiltonians open as the merged 'Hamiltonian' with the order
    parameters that reproduce them, and every original value keeps its slot."""
    from syncmoss.model_io import load_model_from_path

    load_model_from_path(physics_app, _write_legacy_mdl(tmp_path, name, values))
    pt = physics_app.params_table
    assert _model_name(pt, 1) == 'Hamiltonian'
    assert pt.row_params[1] == 15
    model, p = read_model(physics_app)[0], read_model(physics_app)[1]
    assert model[0] == 'Hamiltonian'
    from syncmoss.constants import number_of_baseline_parameters as NB
    got = [float(v) for v in p[NB:NB + 15]]
    assert got == pytest.approx(list(values) + tail)


def test_thick_non_angle_param_not_force_locked(physics_app):
    """Sanity: the default lock is scoped to the angles, not (e.g.) the thickness T."""
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    assert _fix_cb(pt, 1, 0).isChecked() is False   # T (col 0) stays free
