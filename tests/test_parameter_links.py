"""Typing and starting a parameter link '=[X,Y]' in the parameters table.

A value field takes either a plain number or a link ``=[X,Y]`` ("take Y times
the value of parameter X"). Its validator keeps letters out, but it must NOT be
so strict that a half-written link cannot exist: replacing one of the two
numbers normally means deleting it first, which goes through '=[,Y]' / '=[X,]'.
Those intermediate forms are allowed to be typed and are refused by the existing
pre-start check (empty/unfinished parameter slots) when Fit / Show model runs.

The right-click menu of such a field offers the one step that is awkward to type
by hand — starting a link.
"""
import pytest
from PySide6.QtGui import QValidator

from syncmoss.model_io import split_link_field

pytestmark = [pytest.mark.gui]


def _value_input(pt, row, col):
    return pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(1).widget()


def _state(value_input, text):
    """Validator verdict for ``text`` in a value field (Invalid input is what
    QLineEdit silently drops, so it is the one state the user notices)."""
    return value_input.validator().validate(text, len(text))[0]


# --- split_link_field: what counts as a finished link (no GUI needed) --------

@pytest.mark.parametrize("text, expected", [
    ("=[9,1]", (9.0, 1.0)),
    ("=[9,1.0]", (9.0, 1.0)),
    ("=[9,-0.5]", (9.0, -0.5)),
    (" =[9,1] ", (9.0, 1.0)),
    ("=[,1]", None),     # half-written: source index deleted
    ("=[9,]", None),     # half-written: factor deleted
    ("=[,]", None),
    ("=[9,-]", None),
    ("1.0", None),       # a plain number is not a link
    ("", None),
])
def test_split_link_field(text, expected):
    assert split_link_field(text) == expected


# --- what the value field lets the user type --------------------------------

def test_half_written_links_can_be_typed(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Doublet")
    field = _value_input(pt, 1, 1)          # delta

    for text in ("=[9,1]", "=[,1]", "=[9,]", "=[,]", "=[9,-]", "=[", "-0.5"):
        assert _state(field, text) != QValidator.State.Invalid, text


def test_letters_are_still_rejected(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Doublet")
    field = _value_input(pt, 1, 1)

    for text in ("abc", "=[a,1]", "=[9,x]", "1,2", "=[9,1]]"):
        assert _state(field, text) == QValidator.State.Invalid, text


# --- a half-written link blocks show/fit, using the existing red-field path --

def test_unfinished_link_blocks_run_and_marks_field_red(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Doublet")
    field = _value_input(pt, 2, 1)
    field.setText("=[,1]")                  # source index not typed yet

    slots = pt.get_empty_parameter_slots()
    assert [(s["row"], s["col"], s["reason"]) for s in slots] == [(2, 1, "unfinished link")]

    assert physics_app.check_user_expressions("Fit") is False
    assert "was not started" in physics_app.log.toPlainText()
    assert "=[,1]" in physics_app.log.toPlainText()
    assert "red" in field.styleSheet()


def test_missing_factor_blocks_run(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Doublet")
    _value_input(pt, 2, 1).setText("=[9,]")

    assert [s["reason"] for s in pt.get_empty_parameter_slots()] == ["unfinished link"]
    assert physics_app.check_user_expressions("Fit") is False


def test_finished_link_passes(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Doublet")
    _value_input(pt, 2, 1).setText("=[9,1]")

    assert pt.get_empty_parameter_slots() == []
    assert physics_app.check_user_expressions("Show model") is True


# --- the right-click menu ---------------------------------------------------

def test_context_menu_offers_only_the_link_entry(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Doublet")
    menu = pt.value_context_menu(_value_input(pt, 1, 1))
    try:
        assert [a.text() for a in menu.actions()] == ["link to another parameter"]
        assert menu.actions()[0].isEnabled()
    finally:
        menu.deleteLater()


def test_context_menu_entry_prefills_link_with_cursor_before_comma(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Doublet")
    field = _value_input(pt, 1, 1)
    menu = pt.value_context_menu(field)
    try:
        menu.actions()[0].trigger()
    finally:
        menu.deleteLater()

    assert field.text() == "=[,1]"
    # The cursor sits on the empty source-index half, so typing the number of the
    # target parameter is all that is left to do.
    assert field.cursorPosition() == field.text().index(",")
    field.insert("9")
    assert field.text() == "=[9,1]"


def test_starting_a_link_keeps_the_undo_history(physics_app):
    """Regression: the entry used setText(), which clears a QLineEdit's undo
    stack, so Ctrl+Z could not bring the old value back (it works after a
    hand-typed edit)."""
    pt = physics_app.params_table
    pt.select_model(1, "Doublet")
    field = _value_input(pt, 1, 2)          # epsilon, default '1.0'
    before = field.text()

    pt.start_parameter_link(field)
    assert field.text() == "=[,1]"
    assert field.isUndoAvailable()

    field.undo()
    assert field.text() == before


def test_readonly_field_cannot_start_a_link(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")             # par=1 -> greys out the Sextet's delta
    field = _value_input(pt, 1, 1)
    assert field.isReadOnly()

    menu = pt.value_context_menu(field)
    try:
        assert not menu.actions()[0].isEnabled()
    finally:
        menu.deleteLater()
    pt.start_parameter_link(field)
    assert field.text() != "=[,1]"


def test_free_text_field_keeps_the_standard_menu(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")
    field = pt.expression_value_input(2)    # the PDF string, free prose

    menu = pt.value_context_menu(field)
    try:
        texts = [a.text() for a in menu.actions()]
        assert "link to another parameter" not in texts
        assert any("Paste" in t for t in texts)
    finally:
        menu.deleteLater()
