"""GUI-level tests for the polarized ("thick") models, the Layer marker, the
Distr/Corr placement guard, and the SMS-only fit guard.

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
    pt.select_model(1, 'Sextet_(thick)')
    assert _model_name(pt, 1) == 'Sextet_(thick)'
    assert pt.row_params[1] == 12  # T,d,e,H,L,G,theta_h,phi_h,a+,a-,GH,I1/I3


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


def _cms_dat(tmp_path, name='cms.dat'):
    p = tmp_path / name
    p.write_text("#@GCMS 0.25\n")           # CMS marker
    return str(p)


def _sms_dat(tmp_path, name='sms.dat'):
    p = tmp_path / name
    p.write_text("# no instrumental metadata\n")  # -> resolves to SMS (UI fallback)
    return str(p)


def test_thick_model_fit_guard(physics_app, monkeypatch, tmp_path):
    import syncmoss.syncmoss_main as sm
    # Don't pop up a modal dialog during the test.
    monkeypatch.setattr(sm.QMessageBox, 'warning', staticmethod(lambda *a, **k: None))

    pt = physics_app.params_table
    pt.select_model(1, 'Doublet_(thick)')

    # CMS selected -> blocked (regardless of the files)
    physics_app.MS_fit.setChecked(True)
    assert physics_app.check_polarized_models_method([], "Fit") is False

    # SMS, internal instrumental (read-from-file OFF) -> allowed
    physics_app.SMS_fit.setChecked(True)
    physics_app.use_dat_instrumental_metadata = False
    assert physics_app.check_polarized_models_method([_cms_dat(tmp_path)], "Fit") is True

    # SMS + read-from-file ON, the spectrum is CMS by metadata, no Nbaseline -> blocked
    physics_app.use_dat_instrumental_metadata = True
    assert physics_app.check_polarized_models_method([_cms_dat(tmp_path)], "Fit") is False

    # SMS + read-from-file ON, but the spectrum resolves to SMS -> allowed
    assert physics_app.check_polarized_models_method([_sms_dat(tmp_path)], "Fit") is True


def test_thick_guard_nbaseline_sections(physics_app, monkeypatch, tmp_path):
    import syncmoss.syncmoss_main as sm
    monkeypatch.setattr(sm.QMessageBox, 'warning', staticmethod(lambda *a, **k: None))
    physics_app.SMS_fit.setChecked(True)
    physics_app.use_dat_instrumental_metadata = True
    files = [_cms_dat(tmp_path), _sms_dat(tmp_path)]   # spectrum0=CMS, spectrum1=SMS

    pt = physics_app.params_table
    # Thick in the SMS section (section 1) only -> allowed
    pt.select_model(1, 'Doublet')           # section 0 -> CMS spectrum (no thick)
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Doublet_(thick)')   # section 1 -> SMS spectrum
    assert physics_app.check_polarized_models_method(files, "Fit") is True

    # Thick in the CMS section (section 0) -> blocked
    pt.select_model(1, 'Doublet_(thick)')   # section 0 -> CMS spectrum (thick!)
    pt.select_model(3, 'Doublet')
    assert physics_app.check_polarized_models_method(files, "Fit") is False


def test_no_thick_model_is_never_guarded(physics_app, monkeypatch, tmp_path):
    import syncmoss.syncmoss_main as sm
    monkeypatch.setattr(sm.QMessageBox, 'warning', staticmethod(lambda *a, **k: None))
    pt = physics_app.params_table
    pt.select_model(1, 'Doublet')            # scalar only
    physics_app.MS_fit.setChecked(True)      # even in CMS
    physics_app.use_dat_instrumental_metadata = True
    assert physics_app.check_polarized_models_method([_cms_dat(tmp_path)], "Fit") is True
