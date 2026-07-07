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
    assert pt.row_params[1] == 14  # T,d,e,H,L,G,theta_k,phi_h,A,a+,a-,GH,I1/I3,A_m


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


# Orientation/texture parameters each polarized model carries that are FIXED by
# default, as {param column: label}; the user unticks the box to refine them.
# (theta_k, phi_h replace the former scalar asymmetry and are followed by the
# uniaxial texture parameter A; Hamilton_mc keeps its crystal angles and only
# adds the beam-rotation angle alpha_k, with no texture parameter. The
# Faraday-active models also carry a magnetic polar-order A_m right after A,
# likewise locked by default.)
_THICK_LOCKED_ANGLES = {
    'Doublet':     {5: 'θk, °', 6: 'φh, °', 7: 'A'},
    'Sextet':      {6: 'θk, °', 7: 'φh, °', 8: 'A', 9: 'A_m'},
    'MDGD':        {10: 'θk, °', 11: 'φh, °', 12: 'A', 13: 'A_m'},
    'Relax_MS':    {5: 'θk, °', 6: 'φh, °', 7: 'A'},
    'Relax_2S':    {8: 'θk, °', 9: 'φh, °', 10: 'A', 11: 'A_m'},
    'Hamilton_mc': {11: 'αk, °'},
    'ASM':         {9: 'θk, °', 10: 'φh, °', 11: 'A'},
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


def test_thick_non_angle_param_not_force_locked(physics_app):
    """Sanity: the default lock is scoped to the angles, not (e.g.) the thickness T."""
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    assert _fix_cb(pt, 1, 0).isChecked() is False   # T (col 0) stays free
