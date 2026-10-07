"""The 'par' of a Distr/Corr/Recon row -- which parameter of the component above
it the row takes over.

Three things are checked here:

* the grey frame follows 'par' LIVE, including after an Insert/Delete has moved
  the row (the textChanged lambdas carry the row index the widget was built
  with, which the table must not trust),
* two rows of one chain may not claim the same 'par'. They both write into the
  same slot of the shared parameter block (models.TImod: ``pN[int(p[V])] = X``),
  so only the last one would survive -- silently. Show model / Fit must refuse.
* no link =[X,Y] may point at a parameter a Distr/Corr/Recon sets, nor at the
  text column of one. A link copies one plain number before the model is
  calculated: it took the number left in the grey field, never the
  distribution (and a fit moved that number for the link alone), and a text
  column's slot is a placeholder. Show model / Fit refuse, and say which Corr
  or Distr does what the link was meant to.
"""
import pytest

pytestmark = [pytest.mark.gui]

_GREY = "border: 2px solid grey;"

# Flat indices: the baseline p[0..7], the Sextet of row 1 from p[8], then the
# Distr (5 slots) or Recon (7 slots) of row 2
DELTA, EPS = 9, 10
DISTR_L, DISTR_PDF = 23, 26
RECON_WEIGHTS = 28


def _par_input(pt, row):
    """The 'par' field (column 0) of a Distr/Corr/Recon row."""
    return pt._value_input(row, 0)


def _greyed_columns(pt, row):
    return [col for col in range(6) if _GREY in pt._name_label(row, col).styleSheet()]


def test_par_edit_moves_grey_frame_immediately(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")

    assert _greyed_columns(pt, 1) == [1]          # par defaults to 1 -> delta

    _par_input(pt, 2).setText("3")                # -> H, T
    assert _greyed_columns(pt, 1) == [3]
    assert pt._value_input(1, 3).isReadOnly()
    assert not pt._value_input(1, 1).isReadOnly()  # delta released again


def test_par_edit_still_live_after_insert_moves_the_row(physics_app):
    """Regression: on_value_changed used the row index captured when the widget
    was created, so after an Insert shifted the Distr down it looked at the
    neighbouring row, found no marker model and skipped the refresh -- typing a
    new 'par' changed nothing on screen until a model was re-picked."""
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")

    pt.select_model(1, "Insert")                  # Sextet -> row 2, Distr -> row 3
    pt.select_model(1, "Doublet")
    assert [pt.model_name_at(r) for r in (1, 2, 3)] == ["Doublet", "Sextet", "Distr"]

    _par_input(pt, 3).setText("4")
    assert _greyed_columns(pt, 2) == [4]          # the Sextet's L, mm/s
    assert _greyed_columns(pt, 1) == []           # the Doublet is untouched


def test_par_edit_still_live_after_delete_moves_the_row(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Doublet")
    pt.select_model(2, "Sextet")
    pt.select_model(3, "Distr")

    pt.select_model(1, "Delete")                  # Sextet -> row 1, Distr -> row 2

    _par_input(pt, 2).setText("2")
    assert _greyed_columns(pt, 1) == [2]          # the Sextet's epsilon


def test_chain_groups_by_base_component_across_none_rows(physics_app):
    """Rows left at 'None' are invisible to read_model, so they must not cut a
    chain in half (the fit walks the compacted model list)."""
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")
    pt.select_model(3, "Corr")
    pt.select_model(4, "Doublet")
    pt.select_model(5, "Distr")
    pt.select_model(6, "Distr")
    pt.select_model(6, "None")
    pt.select_model(7, "Recon")

    assert pt.get_distribution_chains() == [(1, [2, 3]), (4, [5, 7])]


def test_duplicate_par_in_one_chain_blocks_run_and_marks_red(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")
    pt.select_model(3, "Corr")
    _par_input(pt, 2).setText("3")
    _par_input(pt, 3).setText("3")

    conflicts = pt.get_conflicting_distr_targets()
    assert len(conflicts) == 1
    assert conflicts[0]['par'] == 3
    assert conflicts[0]['rows'] == [2, 3]
    assert conflicts[0]['base_row'] == 1
    assert conflicts[0]['param'] == 'H, T'

    assert physics_app.check_user_expressions("Show model") is False
    message = physics_app.log.toPlainText()
    assert "par" in message and "H, T" in message
    # Only the duplicates turn red, like any other rejected setting. The first
    # claimant keeps its target, so marking it too would leave the user with a
    # red field to click for no reason once the others are moved.
    assert "red" not in _par_input(pt, 2).styleSheet()
    assert "red" in _par_input(pt, 3).styleSheet()

    # Giving them different targets clears the block.
    _par_input(pt, 3).setText("4")
    assert pt.get_conflicting_distr_targets() == []
    assert physics_app.check_user_expressions("Show model") is True
    assert _greyed_columns(pt, 1) == [3, 4]


def test_only_the_later_duplicates_are_marked(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    for row in (2, 3, 4):
        pt.select_model(row, "Distr" if row == 2 else "Corr")
        _par_input(pt, row).setText("2")

    assert physics_app.check_user_expressions("Fit") is False
    marked = [row for row in (2, 3, 4) if "red" in _par_input(pt, row).styleSheet()]
    assert marked == [3, 4]
    # ...but the message names every row involved, so the clash is unambiguous.
    message = physics_app.log.toPlainText()
    assert all(f"table row {row}" in message for row in (2, 3, 4))


def test_same_par_in_separate_chains_is_allowed(physics_app):
    """Each chain has its own parameter block: two components may both have
    their parameter 2 distributed."""
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")
    pt.select_model(3, "Doublet")
    pt.select_model(4, "Distr")
    _par_input(pt, 2).setText("2")
    _par_input(pt, 4).setText("2")

    assert pt.get_conflicting_distr_targets() == []
    assert physics_app.check_user_expressions("Show model") is True


def test_par_written_as_float_is_the_same_target(physics_app):
    """read_model stores 'par' as a float and TImod indexes with int(), so
    '3.0' and '3' address one slot and must clash."""
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")
    pt.select_model(3, "Corr")
    _par_input(pt, 2).setText("3")
    _par_input(pt, 3).setText("3.0")

    assert [c['par'] for c in pt.get_conflicting_distr_targets()] == [3]
    assert _greyed_columns(pt, 1) == [3]


# --- links cannot follow a distribution --------------------------------------

def _log(app):
    return app.log.toPlainText()


def test_a_link_from_another_component_to_a_distributed_parameter_is_refused(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")                   # par 1 -> delta
    pt.select_model(3, "Doublet")
    pt._value_input(3, 1).setText(f"=[{DELTA},1]")
    pt._value_input(3, 2).setText(f"=[{DELTA},0.5]")

    links = pt.get_links_to_distributions()
    assert [(link['reason'], link['row'], link['col'], link['base_row'], link['source_col'])
            for link in links] == [('distributed', 3, 1, 1, 1), ('distributed', 3, 2, 1, 1)]
    assert physics_app.check_user_expressions("Fit") is False
    assert all("red" in pt._value_input(3, col).styleSheet() for col in (1, 2))
    log = _log(physics_app)
    assert ("give it a Distr of its own instead, with par = 1 and the L, R, Num and "
            "probability density function of table row 2") in log
    assert ("with par = 2, L and R of table row 2 times 0.5, its Num, and its probability "
            "density function with X/0.5 in place of X") in log

    pt._value_input(3, 1).setText("0.1")
    pt._value_input(3, 2).setText("0.2")
    assert physics_app.check_user_expressions("Fit") is True


def test_a_link_within_the_component_is_refused_and_the_corr_offered_works(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")                   # par 1 -> delta
    pt._value_input(1, 2).setText(f"=[{DELTA},2]")      # epsilon = 2 delta

    assert physics_app.check_user_expressions("Show model") is False
    assert ("use a Corr right after table row 2 instead: par = 2, dependency "
            "function 2*X") in _log(physics_app)

    pt._value_input(1, 2).setText("0")
    pt.select_model(3, "Corr")
    pt._value_input(3, 0).setText("2")
    pt._value_input(3, 1).setText("2*X")
    assert physics_app.check_user_expressions("Show model") is True


def test_a_link_to_what_a_corr_sets_carries_its_dependency(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")                   # par 1 -> delta
    pt.select_model(3, "Corr")
    pt._value_input(3, 0).setText("2")            # epsilon ...
    pt._value_input(3, 1).setText("0.1*X")        # ... = 0.1 delta
    pt._value_input(1, 3).setText(f"=[{EPS},10]")       # H = 10 epsilon

    assert physics_app.check_user_expressions("Fit") is False
    assert ("which Corr (table row 3) ties to the distribution — use a Corr right after "
            "table row 3 instead: par = 3, dependency function 10*(0.1*X)") in _log(physics_app)


def test_a_recon_is_never_shared_with_another_component(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Recon")                   # par 1 -> delta
    pt.select_model(3, "Doublet")
    pt._value_input(3, 1).setText(f"=[{DELTA},1]")
    pt._value_input(1, 3).setText(f"=[{DELTA},10]")     # H, in the same component

    assert physics_app.check_user_expressions("Fit") is False
    log = _log(physics_app)
    assert "a Recon cannot be shared with another component: remove the link" in log
    assert "par = 3, dependency function 10*X" in log   # a Corr follows a Recon too


@pytest.mark.parametrize("markers, placeholder, column", [
    (["Distr"], DISTR_PDF, "probability density function"),
    (["Recon"], RECON_WEIGHTS, "weights"),
    (["Distr", "Corr"], DISTR_PDF + 2, "dependency function"),
])
def test_a_link_to_a_text_column_is_refused(physics_app, markers, placeholder, column):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    for row, marker in enumerate(markers, start=2):
        pt.select_model(row, marker)
        _par_input(pt, row).setText(str(row - 1))    # a 'par' of its own
    component = len(markers) + 2
    pt.select_model(component, "Doublet")
    pt._value_input(component, 0).setText(f"=[{placeholder},1]")

    [link] = pt.get_links_to_distributions()
    assert (link['reason'], link['column'], link['row']) == ('placeholder', column, component)
    assert physics_app.check_user_expressions("Show model") is False
    assert (f"to the {column} column of {markers[-1]} (table row {component - 1}), "
            f"which holds no number") in _log(physics_app)


def test_links_to_what_no_distribution_sets_still_work(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Distr")                   # par 1 -> delta
    pt.select_model(3, "Doublet")
    pt._value_input(3, 1).setText(f"=[{EPS},1]")        # epsilon: not distributed
    pt._value_input(3, 2).setText(f"=[{DISTR_L},1]")    # the Distr's L: a number

    assert pt.get_links_to_distributions() == []
    assert physics_app.check_user_expressions("Fit") is True
