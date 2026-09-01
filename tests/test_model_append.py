"""Head-less tests for ADDING a model to the current one.

Two dropdown entries share one implementation
(``model_io.load_model_from_path(..., insert_row=...)``): ``Library`` picks the
file from the internal Library folder, ``Load model`` from a file browser. What
is pinned down here is the part that is easy to break silently — the parameter
references of the added model:

* a link into the baseline (the first ``number_of_baseline_parameters`` slots)
  must come out UNCHANGED, because the destination baseline is the one that
  survives and its parameters keep their indices;
* every other link must be re-indexed onto the rows that were just inserted.
"""
import pytest

from syncmoss.constants import number_of_baseline_parameters
from syncmoss import model_io
from syncmoss.model_io import read_model
from syncmoss.parameters_table import MODEL_OPTIONS

pytestmark = pytest.mark.gui

BASE = number_of_baseline_parameters  # 8


def _param_block(value, lower='', upper='', fix='False'):
    """One .mdl parameter: value, lower, upper, name (unused), fix."""
    return [value, lower, upper, '', fix]


def _write_mdl(tmp_path, name, rows, colors=None):
    """Write a minimal .mdl file.

    ``rows`` is a list of ``(model_name, [values...])``; the first entry must be
    the baseline. Values are the value fields only (bounds empty, not fixed).
    """
    names = [model_name for model_name, _ in rows]
    if colors is None:
        colors = [''] * len(rows)
    lines = ['\t'.join(names), '\t'.join(colors)]
    for _, values in rows:
        fields = []
        for value in values:
            fields.extend(_param_block(value))
        lines.append('\t'.join(fields))
    path = tmp_path / name
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(path)


def _value_text(params_table, row, col):
    row_widget = params_table.row_widgets[row]
    param_widget = row_widget.layout().itemAt(col + 1).widget()
    return param_widget.layout().itemAt(1).widget().text()


def _model_name(params_table, row):
    row_widget = params_table.row_widgets[row]
    return row_widget.layout().itemAt(0).widget().layout().itemAt(1).widget().text()


def _source_doublet_mdl(tmp_path, name='source.mdl'):
    """A baseline + Doublet file whose Doublet links to both baseline and itself.

    Doublet parameter layout: T, delta, epsilon, L, G, theta_k, phi_h, A, G2/G1.
    - delta   -> '=[1,1]'  : baseline slot 1 (Os), must survive verbatim
    - epsilon -> '=[8,2]'  : the Doublet's own T (first component slot), shifts
    """
    baseline_values = ['1', '0', '0', '0', '0', '0', '0', '0']
    doublet_values = ['1', '=[1,1]', '=[8,2]', '0.2', '0', '0', '0', '0', '1']
    return _write_mdl(tmp_path, name,
                      [('baseline', baseline_values), ('Doublet', doublet_values)])


def test_load_model_is_offered_next_to_library():
    # The two entries must sit side by side in the dropdown, and 'Load model'
    # must stay AFTER 'Be' so the Library browser's submodel filter (which
    # slices MODEL_OPTIONS at 'Be') never offers it as a model type.
    assert MODEL_OPTIONS.index('Load model') == MODEL_OPTIONS.index('Library') + 1
    assert MODEL_OPTIONS.index('Load model') > MODEL_OPTIONS.index('Be')


def test_append_keeps_baseline_link_and_shifts_component_link(physics_app, tmp_path):
    w = physics_app
    w.params_table.select_model(1, 'Sextet')  # destination: baseline + Sextet
    path = _source_doublet_mdl(tmp_path)

    model_io.load_model_from_path(w, path, insert_row=2)

    assert 'red' not in w.log.styleSheet().lower(), w.log.toPlainText()
    assert _model_name(w.params_table, 2) == 'Doublet'

    # z = index of the last parameter before row 2 = baseline (8) + Sextet (14) - 1.
    z = BASE + 14 - 1
    # Baseline link: verbatim. Self link (source index 8): shifted to z + 1.
    assert _value_text(w.params_table, 2, 1) == '=[1,1]'
    assert _value_text(w.params_table, 2, 2) == f'=[{z + 1},2]'

    model, p, con1, con2, con3, *_ = read_model(w)
    assert model == ['Sextet', 'Doublet']
    assert len(p) == BASE + 14 + 9
    # The two constraints resolve to the intended sources: baseline slot 1 and
    # the appended Doublet's own first parameter.
    sources = {int(c1): int(c2) for c1, c2 in zip(con1, con2)}
    delta_slot = BASE + 14 + 1
    assert sources[delta_slot] == 1
    assert sources[delta_slot + 1] == z + 1


def test_append_directly_after_baseline_leaves_all_links_alone(physics_app, tmp_path):
    """With nothing between the baseline and the insertion point, nothing shifts."""
    w = physics_app
    path = _source_doublet_mdl(tmp_path)

    model_io.load_model_from_path(w, path, insert_row=1)

    assert 'red' not in w.log.styleSheet().lower(), w.log.toPlainText()
    assert _value_text(w.params_table, 1, 1) == '=[1,1]'
    assert _value_text(w.params_table, 1, 2) == '=[8,2]'


def test_append_keeps_the_current_baseline(physics_app, tmp_path):
    """The added file's baseline is discarded, not merged."""
    w = physics_app
    baseline_before = [_value_text(w.params_table, 0, col) for col in range(BASE)]
    # Source baseline holds values that would be visible if it were applied.
    path = _write_mdl(tmp_path, 'other_baseline.mdl', [
        ('baseline', ['12345'] * BASE),
        ('Singlet', ['1', '0', '0.2', '0']),
    ])

    model_io.load_model_from_path(w, path, insert_row=1)

    assert [_value_text(w.params_table, 0, col) for col in range(BASE)] == baseline_before
    assert _model_name(w.params_table, 1) == 'Singlet'


def test_append_repeated_at_same_row_preserves_order_and_links(physics_app, tmp_path):
    """A second add at the same row must not corrupt the first one's links."""
    w = physics_app
    path = _write_mdl(tmp_path, 'two.mdl', [
        ('baseline', ['1', '0', '0', '0', '0', '0', '0', '0']),
        ('Singlet', ['1', '0', '0.2', '0']),
        # Doublet's delta links to the Singlet's delta (source index 9).
        ('Doublet', ['1', '=[9,1]', '0', '0.2', '0', '0', '0', '0', '1']),
    ])

    model_io.load_model_from_path(w, path, insert_row=1)
    assert [_model_name(w.params_table, r) for r in (1, 2)] == ['Singlet', 'Doublet']
    assert _value_text(w.params_table, 2, 1) == '=[9,1]'

    # Add the same file again in front of the first pair.
    model_io.load_model_from_path(w, path, insert_row=1)
    assert 'red' not in w.log.styleSheet().lower(), w.log.toPlainText()
    assert [_model_name(w.params_table, r) for r in (1, 2, 3, 4)] == [
        'Singlet', 'Doublet', 'Singlet', 'Doublet']
    # New pair sits right after the baseline: its link is unshifted and points at
    # the new Singlet's delta (baseline 8 + 1).
    assert _value_text(w.params_table, 2, 1) == '=[9,1]'
    # The pair pushed down was renumbered by the table's own insert bookkeeping
    # (update_references), by the 4 + 9 slots that were inserted above it. It must
    # still point at ITS OWN Singlet's delta, now at 8 + 4 + 9 + 1.
    pushed_singlet_delta = BASE + 4 + 9 + 1
    assert _value_text(w.params_table, 4, 1) == f'=[{pushed_singlet_delta},1]'
    assert pushed_singlet_delta == 9 + (4 + 9)  # old index shifted by what was inserted


def test_load_model_dropdown_entry_appends_via_file_dialog(physics_app, tmp_path, monkeypatch):
    """The 'Load model' menu entry routes to the shared append path."""
    w = physics_app
    w.params_table.select_model(1, 'Sextet')
    path = _source_doublet_mdl(tmp_path, 'dialog.mdl')
    monkeypatch.setattr(model_io.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (path, 'Model files (*.mdl)')))

    model_btn = w.params_table.row_widgets[2].layout().itemAt(0).widget().layout().itemAt(1).widget()
    w.params_table.select_model_by_button('Load model', model_btn)

    # Routed to the append path, NOT turned into a model row named 'Load model'.
    assert _model_name(w.params_table, 2) == 'Doublet'
    z = BASE + 14 - 1
    assert _value_text(w.params_table, 2, 1) == '=[1,1]'
    assert _value_text(w.params_table, 2, 2) == f'=[{z + 1},2]'
    model, *_ = read_model(w)
    assert model == ['Sextet', 'Doublet']


def test_load_model_menu_item_is_clickable(physics_app, tmp_path, monkeypatch):
    """The green menu widget itself must be wired to the append path.

    'Load model' is in the coloured group, so it is a QWidgetAction wrapping a
    QPushButton rather than a plain QAction -- clicking that button is the only
    path a user has.
    """
    from PySide6.QtWidgets import QWidgetAction

    w = physics_app
    path = _source_doublet_mdl(tmp_path, 'clicked.mdl')
    monkeypatch.setattr(model_io.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (path, '')))

    model_btn = w.params_table.row_widgets[1].layout().itemAt(0).widget().layout().itemAt(1).widget()
    entries = {}
    for action in model_btn.menu().actions():
        widget = action.defaultWidget() if isinstance(action, QWidgetAction) else None
        entries[widget.text() if widget is not None else action.text()] = widget
    assert 'Load model' in entries, sorted(entries)
    assert 'Library' in entries
    assert entries['Load model'] is not None  # coloured -> QWidgetAction

    entries['Load model'].click()
    assert _model_name(w.params_table, 1) == 'Doublet'


def test_load_model_dropdown_entry_cancel_is_a_no_op(physics_app, monkeypatch):
    w = physics_app
    w.params_table.select_model(1, 'Sextet')
    monkeypatch.setattr(model_io.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))

    model_btn = w.params_table.row_widgets[2].layout().itemAt(0).widget().layout().itemAt(1).widget()
    w.params_table.select_model_by_button('Load model', model_btn)

    assert _model_name(w.params_table, 2) == 'None'
    model, *_ = read_model(w)
    assert model == ['Sextet']
