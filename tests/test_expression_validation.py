"""Expression/Distr/Corr validation tests.

A meaningless expression typed into an Expression, Distr (probability density
function) or Corr (dependency function) field must:

* be detected before a fit / show-model starts (the procedure is not started),
* be reported in the log box,
* turn the offending field red until the user clicks into it.
"""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from syncmoss.model_io import validate_user_expressions

pytestmark = [pytest.mark.gui]


def _set_expression_text(window, row, text):
    value_input = window.params_table.expression_value_input(row)
    assert value_input is not None
    value_input.setText(text)
    return value_input


def test_valid_expressions_pass(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Distr")       # defaults: par=1, pdf 'X'
    pt.select_model(3, "Corr")        # defaults: par=1, dependency 'X'
    pt.select_model(4, "Expression")  # default 'p[0]'

    _set_expression_text(physics_app, 4, "sqrt(abs(p[0])) + 1")
    assert validate_user_expressions(physics_app) == []


def test_invalid_expression_detected_per_kind(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Distr")
    pt.select_model(3, "Corr")
    pt.select_model(4, "Expression")

    _set_expression_text(physics_app, 2, "X***2")        # syntax error
    _set_expression_text(physics_app, 3, "nonsense(X)")  # unknown name
    _set_expression_text(physics_app, 4, "fwfe")         # unknown name

    problems = validate_user_expressions(physics_app)
    kinds = sorted(p["kind"] for p in problems)
    assert kinds == ["Corr", "Distr", "Expression"]
    rows = {p["kind"]: p["row"] for p in problems}
    assert rows == {"Distr": 2, "Corr": 3, "Expression": 4}


def test_distr_and_corr_take_the_bare_names_of_the_fit(physics_app):
    """Distr/Corr are checked with the bare numpy names the fit gives them: a
    numpy function outside those is refused, the same one as np.<name> passes."""
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Distr")
    pt.select_model(3, "Corr")

    _set_expression_text(physics_app, 2, "heaviside(X, 1)")
    _set_expression_text(physics_app, 3, "hypot(X, 1)")
    problems = validate_user_expressions(physics_app)
    assert {p["kind"]: p["row"] for p in problems} == {"Distr": 2, "Corr": 3}

    _set_expression_text(physics_app, 2, "np.heaviside(X, 1)")
    _set_expression_text(physics_app, 3, "np.hypot(X, 1)")
    assert validate_user_expressions(physics_app) == []


def test_a_parameter_must_be_written_p_number(physics_app):
    """Every reader that finds the parameters of a text as text (renumbering on
    insert/delete, the one-model expansion, the order check, ...) looks for
    p[<number>]: any other spelling is refused, though Python evaluates it."""
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Distr")
    pt.select_model(3, "Corr")
    pt.select_model(4, "Expression")

    for spelling in ("p [8]", "p[ 8 ]", "p[7+1]", "p[-1]", "max(p)"):
        _set_expression_text(physics_app, 2, f"1 + 0*X*{spelling}")
        _set_expression_text(physics_app, 3, f"X + 0*{spelling}")
        _set_expression_text(physics_app, 4, f"2*{spelling}")
        problems = validate_user_expressions(physics_app)
        assert {p["kind"]: p["row"] for p in problems} == {"Distr": 2, "Corr": 3, "Expression": 4}, spelling
        assert all("p[<number>]" in str(p["error"]) for p in problems), spelling

    # p[<number>] itself, and names that merely contain a p, pass
    _set_expression_text(physics_app, 2, "1 + 0*X*p[8]*pi")
    _set_expression_text(physics_app, 3, "X + 0*power(p[8], 2)")
    _set_expression_text(physics_app, 4, "np.exp(p[8])*p[10]")      # T, L: no distribution sets them
    assert validate_user_expressions(physics_app) == []


def test_invalid_expression_blocks_fit_and_marks_field_red(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Expression")
    value_input = _set_expression_text(physics_app, 2, "fwfe")

    # The shared pre-start check refuses to run and reports to the log box
    assert physics_app.check_user_expressions("Fit") is False
    assert "was not started" in physics_app.log.toPlainText()
    assert "fwfe" in physics_app.log.toPlainText()
    assert "red" in physics_app.log.styleSheet()

    # The offending field is highlighted red ...
    assert "red" in value_input.styleSheet()
    assert value_input.property("expression_error") is True

    # ... until the user clicks into it
    QTest.mouseClick(value_input, Qt.MouseButton.LeftButton)
    assert "red" not in value_input.styleSheet()
    assert not value_input.property("expression_error")


def test_fit_pressed_does_not_start_with_invalid_expression(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Expression")
    _set_expression_text(physics_app, 2, "fwfe")

    physics_app.process_path.setPlainText(repr([physics_app.calibration_path]))
    physics_app.fit_pressed()

    # The procedure did not start: no fitting thread, flag released, log explains
    assert physics_app.inprogress is False
    assert getattr(physics_app, "fitting_thread", None) is None
    assert "was not started" in physics_app.log.toPlainText()


def test_show_model_does_not_start_with_invalid_expression(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Distr")
    _set_expression_text(physics_app, 2, "X***2")

    physics_app.process_path.setPlainText(repr([physics_app.calibration_path]))
    physics_app.showM_pressed()

    assert physics_app.inprogress is False
    assert getattr(physics_app, "show_model_thread", None) is None
    assert "was not started" in physics_app.log.toPlainText()
