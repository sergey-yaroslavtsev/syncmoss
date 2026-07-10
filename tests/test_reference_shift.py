"""p[N] parameter references in Expression/Distr/Corr fields must be
re-indexed on structural table edits (insert / swap / delete) exactly like the
=[X,y] references in ordinary parameter fields.

Regression: update_references() used to shift only =[X,y] and silently left
p[N] tokens pointing at the wrong (or deleted) parameters.
"""
import pytest

pytestmark = [pytest.mark.gui]

# Flattened parameter layout used by the tests below:
#   baseline = 8 params -> indices 0..7
#   Singlet  = 4 params
#   Doublet  = 9 params
_BASE = 8


def _expr_text(window, row):
    return window.params_table.expression_value_input(row).text()


def test_insert_shifts_p_references(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")        # indices 8..11
    pt.select_model(2, "Doublet")        # indices 12..20
    pt.select_model(3, "Expression")
    # Reference the Doublet's second parameter (delta, at index 13).
    physics_app.params_table.expression_value_input(3).setText("p[13]*2 + p[8]")

    # Insert a Singlet in front of the Doublet: everything at index >= 12 moves
    # up by 4. p[13] -> p[17]; p[8] (a Singlet param, < 12) stays put.
    pt.select_model(2, "Insert")
    pt.select_model(2, "Singlet")

    assert _expr_text(physics_app, 4) == "p[17]*2 + p[8]"


def test_swap_shifts_p_references(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")        # indices 8..11
    pt.select_model(2, "Doublet")        # indices 12..20
    pt.select_model(3, "Expression")
    physics_app.params_table.expression_value_input(3).setText("p[13]")

    # Swap the leading Singlet for a Doublet (4 -> 9 params, delta +5).
    # Everything at index >= 8 shifts up by 5, so p[13] -> p[18].
    pt.select_model(1, "Doublet")

    assert _expr_text(physics_app, 3) == "p[18]"


def test_delete_shifts_and_flags_p_references(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")        # indices 8..11
    pt.select_model(2, "Doublet")        # indices 12..20
    pt.select_model(3, "Expression")
    physics_app.params_table.expression_value_input(3).setText("p[20]")

    # Delete the Singlet (indices 8..11 removed, delta -4). p[20] -> p[16].
    pt.select_model(1, "Delete")

    # Expression moved from row 3 to row 2 after the delete.
    assert _expr_text(physics_app, 2) == "p[16]"


def _doublet_delta_input(pt, row):
    # Doublet's delta parameter is column 1 (layout index 2).
    return pt.row_widgets[row].layout().itemAt(2).widget().layout().itemAt(1).widget()


def test_deleting_referenced_param_empties_slot_and_blocks_run(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")        # indices 8..11
    pt.select_model(2, "Doublet")        # indices 12..20
    # Doublet's delta references the Singlet's delta (index 9) via =[X,y].
    _doublet_delta_input(pt, 2).setText("=[9,1.0]")

    # Deleting the Singlet clears the referring field -> empty slot.
    pt.select_model(1, "Delete")         # Doublet now at row 1

    empties = pt.get_empty_parameter_slots()
    assert any(s['model'] == 'Doublet' and s['row'] == 1 for s in empties)

    # An empty slot must block both show-model and fit with a clear message.
    assert physics_app.check_user_expressions("Fit") is False
    assert "empty" in physics_app.log.toPlainText().lower()


def test_full_model_has_no_empty_slots(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Doublet")
    pt.select_model(3, "Expression")
    pt.expression_value_input(3).setText("p[8]")

    assert pt.get_empty_parameter_slots() == []
    assert physics_app.check_user_expressions("Show model") is True


def test_clean_model_leaves_no_stale_param_counts(physics_app):
    """The 'Clean model' double-click clears rows via clear_row_params, which
    must zero row_params in step with resetting the button to 'None'. Regression:
    it reset only the button, so the next show/fit walked the now-empty fields of
    a None row and flagged them as missing (red row) even though the model was
    None. Manual delete never hit this because it pops row_params entirely.
    """
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Doublet")

    physics_app.clean_model()

    # Button and parameter count agree: cleared rows carry no active parameters.
    assert pt.row_params[1] == 0
    assert pt.row_params[2] == 0
    # No phantom empty slots, so show/fit is not blocked.
    assert pt.get_empty_parameter_slots() == []
    assert physics_app.check_user_expressions("Show model") is True


def test_delete_of_referenced_param_leaves_dangling_ref_intact(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")        # indices 8..11
    pt.select_model(2, "Doublet")        # indices 12..20
    pt.select_model(3, "Expression")
    # p[9] points inside the Singlet (deleted), p[20] points past it (survives).
    physics_app.params_table.expression_value_input(3).setText("p[9] + p[20]")

    # Delete the Singlet (indices 8..11, delta -4). The deleted reference must be
    # left untouched (NOT silently shifted to a valid-but-wrong index such as
    # p[5]) so the broken expression stays visible; the surviving p[20] -> p[16].
    pt.select_model(1, "Delete")

    assert _expr_text(physics_app, 2) == "p[9] + p[16]"
