"""The results table has exactly the rows of the result it shows.

It used to be a fixed grid of constants.numro*3 = 150 rows built at start-up --
the row limit of the PARAMETERS table -- and every fill stopped silently at row
150: the components after the 50th were simply not shown. Now fill_table creates
three rows per component of the result, however many there are, and clear_table
(what Show model does: a model is not a fit result) removes them all.

"Take result as model" is checked in the cases a result can come in, since it
reads the stored result back into the parameters table, which keeps its fixed
number of rows.

No fit is run: the tables are filled with synthetic numbers the way a fit fills
them -- what is tested is the table, not the physics.
"""
import numpy as np
import pytest
from PySide6.QtWidgets import QLabel, QMessageBox

from syncmoss import model_io
from syncmoss.constants import numco, numro, number_of_baseline_parameters as NB
from syncmoss.parameters_table import result_value_text

pytestmark = [pytest.mark.gui]

BASELINE_NAMES = ['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr']
SINGLET_NAMES = ['T', 'δ, mm/s', 'L, mm/s', 'G, mm/s']
SEXTET = 14          # parameters of a (polarized) Sextet

# More components than the parameters table has rows: what the old grid cut off
MANY = numro + 10


# --- helpers -----------------------------------------------------------------

def _cell(rt, row, col):
    """Plain text of a cell of the interactive table (widget or spanned item)."""
    return rt._cell_text(rt.interactive_table, row, col)


def _singlets(n, expression_at=None):
    """A result of a baseline and *n* Singlets, as fill_table takes it:
    ``(p, models, colors, names, texts, errors)``.

    With *expression_at*, that component is an Expression ``p[8]*2`` instead
    (one slot, its placeholder). Ns and every T are free (error 0.01), the rest
    fixed (NaN), as in a fit -- and few free parameters keep the correlation
    matrix small.
    """
    models, names, colors, texts = ['baseline'], [BASELINE_NAMES], ['lightgray'], {}
    for component in range(1, n + 1):
        if component == expression_at:
            models.append('Expression')
            names.append(['Expression'])
            texts[component] = 'p[8]*2'
        else:
            models.append('Singlet')
            names.append(SINGLET_NAMES)
        colors.append(['blue', 'red', 'lime'][component % 3])
    p = np.arange(sum(len(row) for row in names), dtype=float) + 0.25
    errors = np.full(len(p), np.nan)
    errors[0] = 0.01
    index = NB
    for model, row in zip(models[1:], names[1:]):
        if model == 'Singlet':
            errors[index] = 0.01
        index += len(row)
    return p, models, colors, names, texts, errors


def _fill(rt, result):
    """fill_table the way a fit calls it."""
    p, models, colors, names, texts, errors = result
    n_free = int(np.sum(~np.isnan(errors)))
    rt.fill_table(p, models, colors, names, np.eye(n_free) * 1e-4, errors,
                  np.where(np.isnan(errors))[0], texts)
    return rt


def _stray_labels(rt):
    """Visible labels drawn in the table that belong to no cell."""
    table = rt.interactive_table
    in_cells = {id(table.cellWidget(r, c)) for r in range(table.rowCount())
                for c in range(table.columnCount()) if table.cellWidget(r, c) is not None}
    return [w.text() for w in table.viewport().findChildren(QLabel)
            if not w.isHidden() and id(w) not in in_cells]


def _spans(rt):
    table = rt.interactive_table
    return [(r, c) for r in range(table.rowCount()) for c in range(table.columnCount())
            if table.columnSpan(r, c) > 1]


def _value_input(pt, row, col):
    return pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(1).widget()


def _fake_fit(app, fitted):
    """Fill the results table as a fit of the current parameters table would.

    *fitted* maps flat index -> fitted value. Fixed exactly like the fit: the
    ticked fix boxes plus links, distribution/expression placeholders.
    """
    pt = app.params_table
    model, p, con1, _, _, _, _, _, NExpr, DistriN, _, ReconN = model_io.read_model(app)
    _, fix = model_io.read_bounds_and_fix(app, len(p))
    fix = np.unique(np.concatenate([fix, con1, DistriN, NExpr, ReconN]).astype(int))
    model_rows = model_io.model_file_rows(app)          # taken at fit start
    params = p.copy()
    for index, value in fitted.items():
        params[index] = value
    errors = np.full(len(p), 0.01)
    errors[fix] = np.nan
    n_free = int(np.sum(~np.isnan(errors)))
    rt = app.results_table
    rt.fill_table(params, pt.get_model_list(), pt.get_current_colors(), pt.get_parameter_names(),
                  np.eye(n_free) * 1e-4, errors, fix, pt.get_expression_texts())
    rt.current_links = pt.get_link_snapshot()
    rt.current_model_rows = model_io.fitted_model_rows(model_rows, params)
    rt.current_chi2 = 1.0
    return params


def _show_model(app):
    """What a finished Show model does to the window (synthetic curves)."""
    A = np.linspace(-5.0, 5.0, 32)
    app.on_show_model_finished(A, np.full_like(A, 1000.0), np.full_like(A, 990.0), [], [],
                               np.array([1000.0] + [0.0] * (NB - 1)), [], False, [])


def _answer_take_result_question(monkeypatch, answer):
    """Press *answer* ('Continue' / 'Cancel') in the 'Result model is different'
    question; returns the list the question texts are collected in."""
    asked = []

    def fake_exec(box):
        asked.append(box.text())
        for button in box.buttons():
            if button.text() == answer:
                button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    return asked


# --- the rows follow the result ----------------------------------------------

def test_no_rows_before_a_result(physics_app):
    rt = physics_app.results_table
    assert rt.interactive_table.rowCount() == 0
    assert rt.buttons == [] and rt.labels == []
    assert rt.render_table_to_image().width() > 0        # nothing to draw, no crash


def test_three_rows_per_component_with_its_names_values_and_errors(physics_app):
    rt = physics_app.results_table
    result = _singlets(2)
    p, models = result[0], result[1]
    _fill(rt, result)

    assert rt.interactive_table.rowCount() == 9
    assert len(rt.buttons) == 9
    assert all(len(row) == numco for row in rt.labels)
    for component, model in enumerate(models):
        assert _cell(rt, 3 * component, 0) == model
    # the second Singlet: names, values (3 decimals), errors (only T is free)
    start = NB + 4
    assert [_cell(rt, 6, col + 1) for col in range(4)] == SINGLET_NAMES
    assert [_cell(rt, 7, col + 1) for col in range(4)] == [f"{v:.3f}" for v in p[start:start + 4]]
    assert [_cell(rt, 8, col + 1) for col in range(4)] == ["±0.010", "±nan", "±nan", "±nan"]
    # the % column: the two Singlets share the spectrum's area
    shares = [float(_cell(rt, 3 * c + 1, 0).rstrip('%')) for c in (1, 2)]
    assert sum(shares) == pytest.approx(100.0, abs=0.1)
    assert shares[1] > shares[0]                          # T = p[12] > p[8]


def test_more_components_than_the_parameters_table_has_rows(physics_app):
    """The old grid stopped at component numro-1; nothing is cut any more."""
    rt = physics_app.results_table
    result = _singlets(MANY)
    p = result[0]
    _fill(rt, result)

    assert rt.interactive_table.rowCount() == 3 * (MANY + 1)
    last = MANY
    start = NB + 4 * (MANY - 1)
    assert _cell(rt, 3 * last, 0) == 'Singlet'
    assert [_cell(rt, 3 * last, col + 1) for col in range(4)] == SINGLET_NAMES
    assert [_cell(rt, 3 * last + 1, col + 1) for col in range(4)] == \
        [f"{v:.3f}" for v in p[start:start + 4]]
    assert _cell(rt, 3 * last + 2, 1) == "±0.010"
    shares = [float(_cell(rt, 3 * c + 1, 0).rstrip('%')) for c in range(1, MANY + 1)]
    assert sum(shares) == pytest.approx(100.0, abs=0.05 * MANY)
    # and the whole of it is in the picture and in a copy
    content = rt._rows_with_content()
    assert content == list(range(3 * (MANY + 1)))
    height = sum(rt.interactive_table.rowHeight(row) for row in content)
    assert rt.render_table_to_image().height() == int(1.03 * height)
    rt.interactive_table.selectAll()
    assert f"{p[-1]:.3f}" in rt.copy_selection(rt.interactive_table)


def test_correlation_matrix_has_every_free_parameter(physics_app):
    rt = physics_app.results_table
    result = _singlets(MANY)
    _fill(rt, result)
    n_free = int(np.sum(~np.isnan(result[5])))
    assert n_free == 1 + MANY
    assert rt.correlation_table.rowCount() == n_free + 2
    assert rt.correlation_table.columnCount() == n_free + 2


def test_a_click_on_any_row_reaches_replot(physics_app, monkeypatch):
    rt = physics_app.results_table
    _fill(rt, _singlets(MANY))
    clicked = []
    monkeypatch.setattr(physics_app, 'replot_result', clicked.append)
    row = 3 * (MANY - 2)                                  # far past the old 150 rows
    rt.buttons[row].click()
    assert clicked == [row]


def test_an_expression_past_the_old_limit_is_spanned(physics_app):
    rt = physics_app.results_table
    component = MANY - 3
    result = _singlets(MANY, expression_at=component)
    p = result[0]
    _fill(rt, result)
    rows = [3 * component, 3 * component + 1, 3 * component + 2]
    assert [rt.interactive_table.columnSpan(row, 1) for row in rows] == [numco] * 3
    assert _cell(rt, rows[0], 1) == 'p[8]*2'
    assert _cell(rt, rows[1], 1) == f"{2 * p[8]:.3f}"
    assert _cell(rt, rows[2], 1) == "±0.020"             # 2 x the error of p[8]


def test_a_smaller_result_leaves_nothing_of_a_larger_one(physics_app):
    rt = physics_app.results_table
    big = _singlets(MANY, expression_at=MANY - 3)
    small = _singlets(1)
    for result in (big, small, big, small):
        _fill(rt, result)
        components = len(result[1])
        assert rt.interactive_table.rowCount() == 3 * components
        assert len(rt.buttons) == len(rt.labels) == 3 * components
        assert _stray_labels(rt) == []
    # the small one, and nothing else
    assert _spans(rt) == []
    p = small[0]
    assert [_cell(rt, 4, col + 1) for col in range(4)] == [f"{v:.3f}" for v in p[NB:NB + 4]]
    assert all(_cell(rt, 4, col) == '' for col in range(5, numco + 1))


def test_clear_removes_every_row(physics_app):
    rt = physics_app.results_table
    _fill(rt, _singlets(3))
    rt.clear_table()
    assert rt.interactive_table.rowCount() == 0
    assert rt.buttons == [] and rt.labels == []
    assert rt.correlation_table.rowCount() == rt.correlation_table.columnCount() == 0
    assert rt.render_table_to_image().width() > 0


def test_show_model_empties_the_results_table(physics_app):
    rt = physics_app.results_table
    _fill(rt, _singlets(3))
    _show_model(physics_app)
    assert rt.interactive_table.rowCount() == 0
    assert rt.current_parameters is None
    assert physics_app.inprogress is False
    assert "red" not in physics_app.log.styleSheet().lower(), physics_app.log.toPlainText()


# --- Take result as model ------------------------------------------------------

def test_take_result_fills_the_largest_model_the_table_holds(physics_app):
    """numro-1 Singlets: 150 result rows, every one of them back in the table."""
    pt = physics_app.params_table
    n = numro - 1
    for row in range(1, numro):
        pt.select_model(row, 'Singlet')
    fitted = {}
    for k in range(n):
        start = NB + 4 * k
        fitted[start] = 0.5 + 0.001 * k                  # T
        fitted[start + 1] = -0.25 + 0.01 * k             # delta
        fitted[start + 3] = 0.2                          # G
    _fake_fit(physics_app, fitted)
    assert physics_app.results_table.interactive_table.rowCount() == 3 * numro

    physics_app.take_result()
    for row in (1, n // 2, n):
        start = NB + 4 * (row - 1)
        for col in (0, 1, 3):
            assert _value_input(pt, row, col).text() == \
                result_value_text(fitted[start + col], False), (row, col)
    assert "red" not in physics_app.log.styleSheet().lower(), physics_app.log.toPlainText()


def test_take_result_puts_back_every_section_of_an_nbaseline_result(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    pt.select_model(3, 'Doublet')
    nbaseline = NB + SEXTET
    doublet = nbaseline + NB
    fitted = {0: 1234.5, NB + 1: 0.1, nbaseline: 2345.25, doublet + 1: -0.2, doublet + 2: 0.75}
    _fake_fit(physics_app, fitted)

    rt = physics_app.results_table
    assert rt.interactive_table.rowCount() == 3 * 4
    assert _cell(rt, 6, 0) == 'Nbaseline'
    # each spectrum's components make 100 % of that spectrum
    assert _cell(rt, 4, 0) == '100.0%'
    assert _cell(rt, 10, 0) == '100.0%'

    physics_app.take_result()
    assert [_value_input(pt, *where).text() for where in
            [(0, 0), (1, 1), (2, 0), (3, 1), (3, 2)]] == \
        ['1234.5', '0.1', '2345.25', '-0.2', '0.75']
    assert pt.get_model_list() == ['baseline', 'Sextet', 'Nbaseline', 'Doublet']


def test_take_result_keeps_expression_and_link(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, 'p[9]*2', '', '', 'False')
    slot = int(model_io.read_model(physics_app)[8][0])  # the Expression's slot
    model_io._set_row_param(pt.row_widgets[1], 2, f'=[{slot},1]', '', '', 'False')
    _fake_fit(physics_app, {NB + 1: 0.3, slot: 0.6, NB + 2: 0.6})

    rt = physics_app.results_table
    assert _cell(rt, 6, 1) == 'p[9]*2'
    assert _cell(rt, 7, 1) == '0.600'

    physics_app.take_result()
    assert _value_input(pt, 1, 1).text() == '0.3'
    assert _value_input(pt, 1, 2).text() == f'=[{slot},1]'
    assert _value_input(pt, 2, 0).text() == 'p[9]*2'


def test_take_result_after_show_model_changes_nothing(physics_app):
    """Show model clears the result: F8 then must not touch the model."""
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    _fake_fit(physics_app, {NB + 1: 0.3})
    _value_input(pt, 1, 1).setText('0.123')              # edited after the fit
    _show_model(physics_app)
    before = model_io.model_file_rows(physics_app)

    physics_app.take_result()
    assert model_io.model_file_rows(physics_app) == before
    assert "no fitting results" in physics_app.log.toPlainText().lower()


def test_a_result_larger_than_the_parameters_table_is_refused(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    _value_input(pt, 1, 1).setText('0.123')
    before = model_io.model_file_rows(physics_app)
    _fill(physics_app.results_table, _singlets(MANY))

    physics_app.take_result()
    assert model_io.model_file_rows(physics_app) == before
    log = physics_app.log.toPlainText()
    assert f"has {MANY} components" in log and f"at most {numro - 1}" in log


@pytest.mark.parametrize("answer, kept", [('Continue', 'Sextet'), ('Cancel', 'Doublet')])
def test_take_result_after_the_model_was_changed_asks_first(physics_app, monkeypatch, answer, kept):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    _fake_fit(physics_app, {NB + 1: 0.3})
    pt.select_model(1, 'Doublet')                         # the user moved on
    asked = _answer_take_result_question(monkeypatch, answer)

    physics_app.take_result()
    assert len(asked) == 1 and "different" in asked[0]
    assert pt.get_model_list() == ['baseline', kept]
    if kept == 'Sextet':
        assert _value_input(pt, 1, 1).text() == '0.3'
