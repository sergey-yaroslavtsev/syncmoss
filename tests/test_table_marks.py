"""What the parameters table and the log box show the user before a fit.

* A fit does not start from a value outside its own bounds: minimi_hi only
  printed a warning and fitted on from there. Now the log names the value and
  the field turns red, like every other refused input.
* A finished link =[X,Y] is shown on darkorange, so a parameter that follows
  another one is told apart from a fitted one -- and stays so through every
  refresh of the table that used to reset the field's look.
* Switching to CMS turns a baseline Nnr of 0 into =[0,0.67] (0.67 times Ns),
  and back to SMS exactly that link into 0; any other Nnr is left alone.
* Switching the theme writes the new mode into the log box with a color: a
  stylesheet'd widget keeps painting the old background until its stylesheet
  is set again, and that write sets it.
"""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

pytestmark = [pytest.mark.gui]


def _widgets(pt, row, col):
    param_widget = pt.row_widgets[row].layout().itemAt(col + 1).widget()
    bounds = param_widget.layout().itemAt(2).layout()
    return (param_widget.layout().itemAt(1).widget(),
            bounds.itemAt(0).widget(), bounds.itemAt(1).widget())


def _set_mode(window, cms):
    """Check CMS or SMS through the real handler, as a click would."""
    box = window.MS_fit if cms else window.SMS_fit
    box.setChecked(True)
    window.on_ms_sms_changed(box)


# --- a value outside its bounds ----------------------------------------------

def test_value_outside_its_bounds_blocks_the_fit(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    h_value, h_low, h_high = _widgets(pt, 1, 3)
    h_low.setText('30')
    h_high.setText('35')
    h_value.setText('40')

    outside = pt.get_out_of_bounds_parameters()
    assert [(o['row'], o['col'], o['lower'], o['upper']) for o in outside] == [(1, 3, '', '35')]

    assert physics_app.check_values_within_bounds("Fit") is False
    log = physics_app.log.toPlainText()
    assert "was not started" in log
    assert "'H, T' = 40 is above its upper bound 35" in log
    assert "red" in h_value.styleSheet()
    # ... until the user clicks into it
    QTest.mouseClick(h_value, Qt.MouseButton.LeftButton)
    assert "red" not in h_value.styleSheet()


def test_bounds_are_inclusive_and_an_empty_bound_is_none(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    h_value, h_low, h_high = _widgets(pt, 1, 3)
    h_high.setText('35')
    h_value.setText('35')                 # on the bound: allowed
    _widgets(pt, 1, 1)[0].setText('-1000')  # delta: no bounds at all
    assert pt.get_out_of_bounds_parameters() == []
    assert physics_app.check_values_within_bounds("Fit") is True


def test_fixed_and_baseline_values_are_checked_too(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    _widgets(pt, 0, 0)[0].setText('0')    # Ns, lower bound 1
    _widgets(pt, 1, 8)[0].setText('2')    # A (fixed), upper bound 1
    found = {(o['row'], o['col']): (o['lower'], o['upper'])
             for o in pt.get_out_of_bounds_parameters()}
    assert found == {(0, 0): ('1', ''), (1, 8): ('', '1')}


def test_links_and_distribution_targets_are_not_checked(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Distr')           # par = 1: the Sextet's delta, now grey
    d_value, _, d_high = _widgets(pt, 1, 1)
    assert d_value.isReadOnly()
    d_high.setText('1')
    d_value.setText('5')                  # the distribution axis replaces it anyway
    h_value, _, h_high = _widgets(pt, 1, 3)
    h_high.setText('35')
    h_value.setText('=[8,100]')           # a link follows its source
    assert pt.get_out_of_bounds_parameters() == []


@pytest.mark.parametrize("cms", [False, True])
def test_no_model_starts_out_of_its_own_bounds(physics_app, cms):
    """Every row a model menu entry builds is accepted as it comes."""
    _set_mode(physics_app, cms)
    pt = physics_app.params_table
    models = ['Singlet', 'Doublet', 'Sextet', 'MDGD', 'Relax_MS', 'Relax_2S',
              'Hamiltonian', 'ASM', 'SCDW', 'Be', 'KB_nano',
              'Sextet', 'Distr', 'Corr', 'Sextet', 'Recon',
              'Variables', 'Expression', 'Nbaseline']
    for row, model in enumerate(models, start=1):
        pt.select_model(row, model)
    assert pt.get_out_of_bounds_parameters() == []
    _set_mode(physics_app, cms=False)


def test_fit_button_refuses_to_start(physics_app, monkeypatch):
    # Should the check ever let it through, stop before any fit is started
    monkeypatch.setattr(physics_app, "confirm_instrumental_methods", lambda *args: False)
    _widgets(physics_app.params_table, 0, 0)[0].setText('0')     # Ns below 1

    physics_app.process_path.setPlainText(repr([physics_app.calibration_path]))
    physics_app.fit_pressed()

    assert physics_app.inprogress is False
    assert getattr(physics_app, "fitting_thread", None) is None
    assert "outside their bounds" in physics_app.log.toPlainText()


# --- links are shown in darkorange -------------------------------------------

def test_finished_link_is_highlighted(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Doublet')
    field = _widgets(pt, 1, 1)[0]

    field.setText('=[8,1]')
    assert 'darkorange' in field.styleSheet()
    field.setText('=[,1]')                # half-written: not a link yet
    assert field.styleSheet() == ''
    field.setText('0.5')
    assert field.styleSheet() == ''


def test_highlight_survives_table_refreshes(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Singlet')         # p[8..11]
    pt.select_model(2, 'Doublet')
    field = _widgets(pt, 2, 1)[0]
    field.setText('=[9,1]')               # delta of the Doublet = delta of the Singlet

    pt.select_model(3, 'Singlet')         # redraws the grey frames of the whole table
    assert 'darkorange' in field.styleSheet()

    pt.select_model(1, 'Insert')          # renumbers the link ...
    pt.select_model(1, 'Singlet')
    field = _widgets(pt, 3, 1)[0]
    assert field.text() == '=[13,1]'
    assert 'darkorange' in field.styleSheet()   # ... and keeps it orange

    pt.mark_parameter_error(3, 1)         # a refused start reddens it ...
    assert 'red' in field.styleSheet()
    QTest.mouseClick(field, Qt.MouseButton.LeftButton)
    assert 'darkorange' in field.styleSheet()   # ... and a click brings it back


# --- Nnr follows the CMS/SMS switch -------------------------------------------

def test_nnr_follows_the_mode(physics_app):
    nnr = _widgets(physics_app.params_table, 0, 4)[0]
    assert nnr.text() == '0'

    _set_mode(physics_app, cms=True)
    assert nnr.text() == '=[0,0.67]'
    assert 'darkorange' in nnr.styleSheet()
    _set_mode(physics_app, cms=False)
    assert nnr.text() == '0'

    nnr.setText('0.0')                    # zero written another way
    _set_mode(physics_app, cms=True)
    assert nnr.text() == '=[0,0.67]'
    _set_mode(physics_app, cms=False)


@pytest.mark.parametrize("cms, text", [
    (True, '5'),                          # a real Nnr is the user's
    (False, '=[0,0.5]'),                  # so is another link
])
def test_other_nnr_values_are_kept(physics_app, cms, text):
    _set_mode(physics_app, cms=not cms)
    nnr = _widgets(physics_app.params_table, 0, 4)[0]
    nnr.setText(text)
    _set_mode(physics_app, cms=cms)
    assert nnr.text() == text
    _set_mode(physics_app, cms=False)


# --- the theme switch is written into the log box -----------------------------

def test_theme_switch_rewrites_the_log_box(physics_app):
    w = physics_app
    w.set_status("something earlier", "red")
    w.toggle_theme()
    try:
        assert w.log.toPlainText() == f"Switched to {w._theme['name']}"
        # set again, with the new mode's plain text color
        assert w.log.styleSheet() == f"color: {'white' if w._is_dark_mode else 'black'};"
    finally:
        w.toggle_theme()
