"""The "Take result as model" button puts the fitted model back into the table.

Two things went wrong with that:

* the bounds the user had set vanished. take_result rebuilds every row through
  select_model, which fills in the model's DEFAULT bounds, so any other bound
  was lost -- most visibly on a parameter that stopped on such a bound, which
  then came back fixed (its error is NaN) without the bound that stopped it.
  The bounds are now taken from the model the result was fitted with.
* the values were written with '%.6g': '1.23457e+06', '1.5e-05', six digits
  after the point. A fitted value now gets at most four decimals and never an
  exponent; a value the fit did not move keeps every digit (a typed fixed value,
  a preset row, a value stopped on a bound), and so does one that rounding would
  push across its own bound -- the next fit would be refused for starting out of
  bounds.

No fit is run: the results table is filled the way a fit fills it.
"""
import numpy as np
import pytest

from syncmoss import model_io
from syncmoss.parameters_table import format_parameter_value, result_value_text

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


# --- the text a value is written with (no GUI needed) ------------------------

@pytest.mark.parametrize("value, text", [
    (1234567.891234, "1234567.8912"),   # '%.6g' gave '1.23457e+06'
    (1.5e-05, "0"),                     # '%.6g' gave '1.5e-05'
    (-1.5e-05, "0"),                    # no '-0'
    (0.123456789, "0.1235"),
    (33.0, "33"),
    (0.098, "0.098"),
    (1e16, "10000000000000000"),
])
def test_four_decimals_and_never_an_exponent(value, text):
    assert format_parameter_value(value) == text


def test_every_digit_is_kept_on_request_still_without_exponent():
    assert format_parameter_value(0.987654321, decimals=None) == "0.987654321"
    assert format_parameter_value(1.5e-05, decimals=None) == "0.000015"


def test_rounding_never_crosses_a_bound():
    # 0.12345 rounds to 0.1235, above an upper bound of 0.12346: keep it whole
    assert result_value_text(0.12345, False, '', '0.12346') == "0.12345"
    assert result_value_text(0.12345, False, '0.1234', '') == "0.1235"
    assert result_value_text(0.987654321, True) == "0.987654321"


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


def test_fitted_values_are_short_fixed_ones_exact(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    _widgets(pt, 1, 8)[0].setText('0.987654321')     # A, fixed by default
    _widgets(pt, 1, 5)[2].setText('0.12346')         # G: an upper bound with 5 decimals

    _fake_fit(physics_app, {NS: 1234567.891234, T: 1.5e-05, DELTA: 0.123456789,
                            EPS: -0.0000123, G: 0.12345})
    physics_app.take_result()

    texts = {index: _widgets(pt, *where)[0].text() for index, where in
             {NS: (0, 0), T: (1, 0), DELTA: (1, 1), EPS: (1, 2), G: (1, 5), A: (1, 8)}.items()}
    assert texts == {NS: '1234567.8912', T: '0', DELTA: '0.1235', EPS: '0',
                     G: '0.12345', A: '0.987654321'}
    for row in range(len(pt.row_widgets)):
        for col in range(pt.row_params[row]):
            assert 'e' not in _widgets(pt, row, col)[0].text().lower()
    # ... so the next fit can start right away
    assert pt.get_out_of_bounds_parameters() == []


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
