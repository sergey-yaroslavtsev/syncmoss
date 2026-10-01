"""The results-table picture (the table half of _combo.png and the sequence
HTML page) shows exactly what the table shows.

An Expression's single column is spanned across the row with a table ITEM
instead of a label widget. Two things went wrong with that in the picture:

* a check that looks at widgets only took the Expression's value row and its
  ± row for empty rows and hid them: the Expression came out on one line;
* removeCellWidget only unregisters the label it swaps out -- the label stays a
  visible child of the table until Qt deletes it later, and the picture is
  taken before that (right after the fit filled the table), so it drew the old
  label under the item: the texts on top of each other, one more layer per fit.
"""
import numpy as np
import pytest
from PySide6.QtWidgets import QLabel

from syncmoss import model_io

pytestmark = [pytest.mark.gui]


def _model(app):
    """baseline + Sextet + Expression 'p[11]*2'."""
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, 'p[11]*2', '', '', 'False')


def _fill(app, shift=0.5):
    """Fill the results table the way a fit does; returns it."""
    pt = app.params_table
    n = len(model_io.read_model(app)[1])
    rt = app.results_table
    rt.fill_table(np.arange(n, dtype=float) + shift, pt.get_model_list(), pt.get_current_colors(),
                  pt.get_parameter_names(), np.eye(n) * 1e-4, np.full(n, 0.01),
                  np.array([], dtype=int), pt.get_expression_texts())
    return rt


def _stray_labels(rt):
    """Labels drawn in the table that belong to no cell (left behind by a swap)."""
    table = rt.interactive_table
    in_cells = {id(table.cellWidget(r, c)) for r in range(table.rowCount())
                for c in range(table.columnCount()) if table.cellWidget(r, c) is not None}
    return [w.text() for w in table.viewport().findChildren(QLabel)
            if not w.isHidden() and id(w) not in in_cells]


def test_all_three_rows_of_an_expression_are_in_the_picture(physics_app):
    _model(physics_app)
    rt = _fill(physics_app)
    expression = [6, 7, 8]                  # component 2: name, value and ± rows
    assert set(expression) <= set(rt._rows_with_content())
    texts = [rt.interactive_table.item(row, 1).text() for row in expression]
    assert texts[0] == 'p[11]*2'
    assert texts[1] == f"{2 * 11.5:.3f}"    # the value, on its own row
    assert texts[2].startswith('±')         # and its deviation on the next


def test_the_picture_is_as_tall_as_the_rows_it_shows(physics_app):
    _model(physics_app)
    rt = _fill(physics_app)
    rows = rt._rows_with_content()
    height = sum(rt.interactive_table.rowHeight(row) for row in rows)
    assert rt.render_table_to_image().height() == int(1.03 * height)
    assert len(rows) == 9                   # baseline, Sextet and Expression: 3 rows each


def test_no_swapped_out_label_is_left_under_an_expression(physics_app):
    """Fill after fill, as a sequence does: nothing but the cells is drawn."""
    _model(physics_app)
    for shift in (0.5, 1.5, 2.5):
        rt = _fill(physics_app, shift)
        assert _stray_labels(rt) == []
