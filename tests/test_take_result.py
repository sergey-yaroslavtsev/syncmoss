"""The "Take result as model" button puts the fitted model back into the table.

Two things went wrong with that:

* the bounds the user had set vanished. take_result rebuilds every row through
  select_model, which fills in the model's DEFAULT bounds, so any other bound
  was lost -- most visibly on a parameter that stopped on such a bound, which
  then came back fixed (its error is NaN) without the bound that stopped it.
  The bounds are now taken from the model the result was fitted with.
* the values were written with '%.6g' ('1.23457e+06', '1.5e-05'), later with
  four decimals, which moved every fitted value a little. Now every digit goes
  in, never with an exponent, and the value field only SHOWS three decimals
  (parameters_table.ValueLineEdit): the next fit, a saved model and Copy /
  Paste carry on from exactly the fitted values.

No fit is run: the results table is filled the way a fit fills it.
"""
import numpy as np
import pytest

from syncmoss import model_io
from syncmoss.parameters_table import format_parameter_value, shown_value_text
from syncmoss.syncmoss_main import _recon_weight_texts

pytestmark = [pytest.mark.gui]

# Flat indices: baseline p[0..7], then the Sextet of row 1 from p[8]
NS = 0
T, DELTA, EPS, H, L, G, A = 8, 9, 10, 11, 12, 13, 16


def _widgets(pt, row, col):
    param_widget = pt.row_widgets[row].layout().itemAt(col + 1).widget()
    bounds = param_widget.layout().itemAt(2).layout()
    return (param_widget.layout().itemAt(1).widget(),
            bounds.itemAt(0).widget(), bounds.itemAt(1).widget(),
            param_widget.layout().itemAt(0).layout().itemAt(1).widget())


def _fake_fit(app, fitted, stopped_on_bound=()):
    """Fill the results table as a fit of the current table would.

    *fitted* maps flat index -> fitted value; *stopped_on_bound* lists free
    parameters minimi_hi fixed on a bound (their error comes back NaN).
    """
    pt = app.params_table
    p = model_io.read_model(app)[1]
    _, fix = model_io.read_bounds_and_fix(app, len(p))
    model_rows = model_io.model_file_rows(app)          # taken at fit start
    params = p.copy()
    for index, value in fitted.items():
        params[index] = value
    errors = np.full(len(p), 0.01)
    errors[fix] = np.nan
    errors[list(stopped_on_bound)] = np.nan
    n_free = int(np.sum(~np.isnan(errors)))
    rt = app.results_table
    rt.fill_table(params, pt.get_model_list(), pt.get_current_colors(), pt.get_parameter_names(),
                  np.eye(n_free) * 1e-4, errors, fix, pt.get_expression_texts())
    rt.current_links = pt.get_link_snapshot()
    rt.current_model_rows = model_io.fitted_model_rows(model_rows, params)


# --- the text a value is shown with (no GUI needed) --------------------------

@pytest.mark.parametrize("value, text", [
    (1234567.891234, "1234567.891"),    # '%.6g' gave '1.23457e+06'
    (1.5e-05, "0"),                     # '%.6g' gave '1.5e-05'
    (-1.5e-05, "0"),                    # no '-0'
    (0.123456789, "0.123"),
    (33.0, "33"),
    (0.098, "0.098"),
    (1e16, "10000000000000000"),
])
def test_three_decimals_and_never_an_exponent(value, text):
    assert format_parameter_value(value) == text


def test_every_digit_is_kept_on_request_still_without_exponent():
    assert format_parameter_value(0.987654321, decimals=None) == "0.987654321"
    assert format_parameter_value(1.5e-05, decimals=None) == "0.000015"


@pytest.mark.parametrize("text, shown", [
    ("0.123456789", "0.123"),
    ("1.5e-05", "0"),                       # an exponent is never shown
    ("=(0.123456)", "=(0.123)"),            # the X of an independent value
    ("1.0", "1.0"),                         # short enough: as it is
    ("=(0.05)", "=(0.05)"),
    ("=[9,0.123456]", "=[9,0.123456]"),     # a link as it is
    ("=[,1]", "=[,1]"),
    ("", ""),
])
def test_what_a_value_field_shows(text, shown):
    assert shown_value_text(text) == shown


# --- the value field keeps every digit ---------------------------------------

def test_a_value_field_keeps_every_digit_and_shows_three_decimals(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    field = _widgets(pt, 1, 1)[0]
    field.setText('0.123456789')
    assert (field.text(), field.displayText()) == ('0.123456789', '0.123')
    assert field.toolTip() == '0.123456789'                  # the hidden digits
    assert model_io.read_model(physics_app)[1][DELTA] == 0.123456789
    assert model_io.model_file_rows(physics_app)[2][1][1][0] == '0.123456789'

    pt.copy_model_to_memory(1)
    pt.paste_model_from_memory(2)
    assert _widgets(pt, 2, 1)[0].text() == '0.123456789'

    pt.select_model(3, 'Expression')                         # free text: as it is
    expression = _widgets(pt, 3, 0)[0]
    expression.setText('0.123456789')
    assert expression.displayText() == '0.123456789'


def test_an_edit_makes_the_typed_text_the_value_and_undo_brings_the_digits_back(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    field = _widgets(pt, 1, 1)[0]
    field.setText('0.123456789')
    field.insert('4')                                        # typed after '0.123'
    assert field.text() == '0.1234' and field.toolTip() == ''
    field.undo()
    assert field.text() == '0.123456789' and field.toolTip() == '0.123456789'


def test_a_saved_model_loads_back_with_every_digit(physics_app, tmp_path):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    _widgets(pt, 1, 1)[0].setText('0.123456789')
    path = str(tmp_path / 'model.mdl')
    assert model_io._save_model_to_file(physics_app, path)
    pt.select_model(1, 'Doublet')
    model_io.load_model_from_path(physics_app, path)
    field = _widgets(pt, 1, 1)[0]
    assert (field.text(), field.displayText()) == ('0.123456789', '0.123')


# --- the whole round trip ----------------------------------------------------

def test_user_bounds_survive_a_parameter_stopping_on_them(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    h_value, h_low, h_high, h_fix = _widgets(pt, 1, 3)
    h_low.setText('30')                       # H has no default bounds
    h_high.setText('35')
    _, d_low, d_high, _ = _widgets(pt, 1, 1)
    d_low.setText('-1')
    d_high.setText('1')
    assert not h_fix.isChecked()

    _fake_fit(physics_app, {H: 35.0, DELTA: 0.123456789}, stopped_on_bound=[H])
    physics_app.take_result()

    assert (h_low.text(), h_high.text()) == ('30', '35')
    assert h_value.text() == '35'
    assert h_fix.isChecked()                  # stopped on the bound -> fixed, as before
    assert (d_low.text(), d_high.text()) == ('-1', '1')
    # the default bounds of the other parameters are still there
    _, l_low, _, _ = _widgets(pt, 1, 4)
    assert l_low.text() == '0.098'


def test_fitted_values_keep_every_digit_and_show_three_decimals(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    _widgets(pt, 1, 8)[0].setText('0.987654321')     # A, fixed by default
    _widgets(pt, 1, 5)[2].setText('0.12346')         # G: an upper bound with 5 decimals

    _fake_fit(physics_app, {NS: 1234567.891234, T: 1.5e-05, DELTA: 0.123456789,
                            EPS: -0.0000123, G: 0.12345})
    physics_app.take_result()

    where = {NS: (0, 0), T: (1, 0), DELTA: (1, 1), EPS: (1, 2), G: (1, 5), A: (1, 8)}
    fields = {index: _widgets(pt, *at)[0] for index, at in where.items()}
    assert {index: field.text() for index, field in fields.items()} == {
        NS: '1234567.891234', T: '0.000015', DELTA: '0.123456789', EPS: '-0.0000123',
        G: '0.12345', A: '0.987654321'}
    assert {index: field.displayText() for index, field in fields.items()} == {
        NS: '1234567.891', T: '0', DELTA: '0.123', EPS: '0', G: '0.123', A: '0.988'}
    for row in range(len(pt.row_widgets)):
        for col in range(pt.row_params[row]):
            field = _widgets(pt, row, col)[0]
            assert 'e' not in field.text().lower() + field.displayText().lower()
    # ... so the next fit starts from exactly the fitted values
    assert np.array_equal(model_io.read_model(physics_app)[1][[NS, T, DELTA, EPS, G]],
                          [1234567.891234, 1.5e-05, 0.123456789, -0.0000123, 0.12345])
    assert pt.get_out_of_bounds_parameters() == []


def test_fitted_recon_weights_go_back_with_every_digit():
    weights = np.array([0.123456789012, 1e-9, 2.0])
    texts = _recon_weight_texts(['baseline', 'Sextet', 'Recon'], [weights])
    assert list(texts) == [2]
    assert np.array_equal(model_io.parse_recon_weights(texts[2], 3), weights)


def test_a_preset_keeps_its_exact_digits(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Be')
    be_slice = slice(8 + 14, 8 + 14 + 9)
    before = model_io.read_model(physics_app)[1][be_slice]

    _fake_fit(physics_app, {DELTA: 0.2})
    physics_app.take_result()

    # bit-identical, so the results table still recognises it as the impurity
    assert np.array_equal(model_io.read_model(physics_app)[1][be_slice], before)
