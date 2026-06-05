"""
Library dialog windows.

Builds the PySide6 dialogs for saving the current model to the Library (with
metadata) and for browsing/filtering existing library models, then inserting a
chosen model into the parameters table.
"""
import os
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from syncmoss.constants import numco
from syncmoss.Library_io import LIBRARY_METADATA_FIELDS, parse_library_model_file, library_model_sort_key
from syncmoss.model_io import load_model_from_path, read_model, save_model_to_library


def _set_log(main_window, message, color):
    main_window.log.setPlainText(message)
    main_window.log.setStyleSheet(f"color: {color};")


def save_to_library_via_dialog(main_window):
    """Ask for title/comment/metadata and save model into internal Library folder."""
    try:
        model, *_ = read_model(main_window)
        if 'Nbaseline' in model:
            _set_log(main_window, "Model with 'Nbaseline' could not be saved to library", "orange")
            return
    except Exception as e:
        _set_log(main_window, f"Could not validate model before saving: {e}", "red")
        return

    dialog = QDialog(main_window)
    dialog.setWindowTitle("Save to library")
    dialog.setModal(True)
    layout = QVBoxLayout(dialog)

    title_label = QLabel("Title:")
    title_input = QLineEdit(dialog)
    title_input.setPlaceholderText("Model name")

    composition_label = QLabel("Chemical composition:")
    composition_input = QLineEdit(dialog)
    composition_input.setPlaceholderText("e.g. Li3Fe2O5")

    temperature_label = QLabel("Temperature (K):")
    temperature_input = QLineEdit(dialog)
    temperature_input.setText("300")

    pressure_label = QLabel("Pressure (GPa):")
    pressure_input = QLineEdit(dialog)
    pressure_input.setText("0.00001")

    field_label = QLabel("External field (T):")
    field_input = QLineEdit(dialog)
    field_input.setText("0")

    doi_label = QLabel("DOI:")
    doi_input = QLineEdit(dialog)
    doi_input.setPlaceholderText("e.g. 10.1000/xyz123")

    comment_label = QLabel("Comment:")
    comment_input = QTextEdit(dialog)
    comment_input.setPlaceholderText("Optional comment")
    comment_input.setMaximumHeight(120)

    layout.addWidget(title_label)
    layout.addWidget(title_input)
    layout.addWidget(composition_label)
    layout.addWidget(composition_input)
    layout.addWidget(temperature_label)
    layout.addWidget(temperature_input)
    layout.addWidget(pressure_label)
    layout.addWidget(pressure_input)
    layout.addWidget(field_label)
    layout.addWidget(field_input)
    layout.addWidget(doi_label)
    layout.addWidget(doi_input)
    layout.addWidget(comment_label)
    layout.addWidget(comment_input)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        _set_log(main_window, "Saving to library canceled", "orange")
        return

    title = title_input.text().strip()
    comment = comment_input.toPlainText().strip()
    metadata = {
        'Chemical composition': composition_input.text().strip(),
        'Temperature (K)': temperature_input.text().strip() or '300',
        'Pressure (GPa)': pressure_input.text().strip() or '0.00001',
        'External field (T)': field_input.text().strip() or '0',
        'DOI': doi_input.text().strip(),
    }
    save_model_to_library(main_window, title, comment if comment else None, metadata=metadata, notify_rename=True)


def open_library_model_dialog(main_window, parent_widget, insert_row, model_options=None):
    """Open internal Library browser with filters and detailed preview."""
    library_dir = os.path.join(main_window.dir_path, 'Library')
    if not os.path.isdir(library_dir):
        _set_log(main_window, f"Library folder not found: {library_dir}", "red")
        return

    model_files = sorted(
        [
            f for f in os.listdir(library_dir)
            if os.path.isfile(os.path.join(library_dir, f))
        ],
        key=library_model_sort_key,
    )
    if not model_files:
        _set_log(main_window, "Library folder is empty", "orange")
        return

    parsed_items = []
    for file_name in model_files:
        path = os.path.join(library_dir, file_name)
        try:
            parsed_items.append(parse_library_model_file(path))
        except Exception as e:
            _set_log(main_window, f"Could not parse library model {file_name}: {e}", "orange")

    if not parsed_items:
        _set_log(main_window, "No readable model files in Library", "orange")
        return

    dialog = QDialog(parent_widget)
    dialog.setWindowTitle("Library")
    dialog.resize(1300, 760)
    root_layout = QVBoxLayout(dialog)

    content_layout = QHBoxLayout()
    root_layout.addLayout(content_layout)

    # Left panel: filters
    filters_widget = QWidget(dialog)
    filters_layout = QVBoxLayout(filters_widget)
    filters_layout.addWidget(QLabel("Filters"))

    grid = QGridLayout()
    lbl_temp = QLabel("Temperature (K) from")
    lbl_temp.setStyleSheet("font-weight: bold;")
    grid.addWidget(lbl_temp, 0, 0)
    temp_from = QLineEdit(filters_widget)
    temp_to = QLineEdit(filters_widget)
    grid.addWidget(temp_from, 0, 1)
    grid.addWidget(QLabel("to"), 0, 2)
    grid.addWidget(temp_to, 0, 3)

    lbl_press = QLabel("Pressure (GPa) from")
    lbl_press.setStyleSheet("font-weight: bold;")
    grid.addWidget(lbl_press, 1, 0)
    press_from = QLineEdit(filters_widget)
    press_to = QLineEdit(filters_widget)
    grid.addWidget(press_from, 1, 1)
    grid.addWidget(QLabel("to"), 1, 2)
    grid.addWidget(press_to, 1, 3)

    lbl_field = QLabel("External field (T) from")
    lbl_field.setStyleSheet("font-weight: bold;")
    grid.addWidget(lbl_field, 2, 0)
    field_from = QLineEdit(filters_widget)
    field_to = QLineEdit(filters_widget)
    grid.addWidget(field_from, 2, 1)
    grid.addWidget(QLabel("to"), 2, 2)
    grid.addWidget(field_to, 2, 3)

    lbl_comp = QLabel("Chemical composition")
    lbl_comp.setStyleSheet("font-weight: bold;")
    grid.addWidget(lbl_comp, 3, 0)
    composition_filter = QLineEdit(filters_widget)
    composition_filter.setPlaceholderText("e.g. Fe Li or exact match e.g. Fe2O3")
    grid.addWidget(composition_filter, 3, 1, 1, 3)

    lbl_title = QLabel("Title")
    lbl_title.setStyleSheet("font-weight: bold;")
    grid.addWidget(lbl_title, 4, 0)
    title_filter = QLineEdit(filters_widget)
    title_filter.setPlaceholderText("e.g. borate iron")
    grid.addWidget(title_filter, 4, 1, 1, 3)

    lbl_sub = QLabel("Submodel")
    lbl_sub.setStyleSheet("font-weight: bold;")
    grid.addWidget(lbl_sub, 5, 0)
    submodel_combo = QComboBox(filters_widget)
    submodel_combo.addItem("(any)")
    submodel_names = []
    if model_options:
        try:
            asm_index = model_options.index('ASM')
            submodel_names = list(model_options[:asm_index + 1])
        except ValueError:
            submodel_names = list(model_options)
    else:
        # Fallback: keep current behavior if options are not provided.
        submodel_names = sorted({m for item in parsed_items for m in item.get('submodel_counts', {}).keys()})
    for name in submodel_names:
        if name and name != 'None':
            submodel_combo.addItem(name)
    submodel_count = QSpinBox(filters_widget)
    submodel_count.setMinimum(1)
    submodel_count.setMaximum(999)
    submodel_count.setValue(1)
    submodel_count.setEnabled(False)
    grid.addWidget(submodel_combo, 5, 1, 1, 2)
    grid.addWidget(submodel_count, 5, 3)

    filters_layout.addLayout(grid)
    filters_layout.addStretch(1)
    filters_widget.setMinimumWidth(320)

    # Middle panel: list
    list_widget = QListWidget(dialog)
    list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

    # Right panel: metadata + model preview table
    right_widget = QWidget(dialog)
    right_layout = QVBoxLayout(right_widget)
    right_layout.addWidget(QLabel("Model information"))
    info_label = QLabel(right_widget)
    info_label.setTextFormat(Qt.TextFormat.RichText)
    info_label.setWordWrap(True)
    info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    info_label.setMinimumHeight(180)
    right_layout.addWidget(info_label)

    preview_table = QTableWidget(right_widget)
    preview_table.setColumnCount(numco + 1)
    preview_table.horizontalHeader().setVisible(False)
    preview_table.verticalHeader().setVisible(False)
    preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    preview_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    preview_table.setAlternatingRowColors(True)
    for c in range(numco + 1):
        preview_table.setColumnWidth(c, 64)
    right_layout.addWidget(preview_table)

    content_layout.addWidget(filters_widget)
    content_layout.addWidget(list_widget, 1)
    content_layout.addWidget(right_widget, 2)

    for info in parsed_items:
        info['list_text'] = info.get('file_name', '')

    def safe_float(text):
        try:
            return float(str(text).strip())
        except Exception:
            return None

    def item_matches_filters(info):
        md = info.get('metadata', {})

        numeric_pairs = [
            ('Temperature (K)', temp_from.text(), temp_to.text()),
            ('Pressure (GPa)', press_from.text(), press_to.text()),
            ('External field (T)', field_from.text(), field_to.text()),
        ]

        for key, lo_txt, hi_txt in numeric_pairs:
            lo = safe_float(lo_txt) if str(lo_txt).strip() else None
            hi = safe_float(hi_txt) if str(hi_txt).strip() else None
            if lo is None and hi is None:
                continue
            val = safe_float(md.get(key, ''))
            if val is None:
                return False
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False

        composition_text = composition_filter.text().strip().lower()
        if composition_text:
            composition = str(md.get('Chemical composition', '')).lower()
            tokens = [t for t in re.split(r'\s+', composition_text) if t]
            for token in tokens:
                if token not in composition:
                    return False

        title_text = title_filter.text().strip().lower()
        if title_text:
            title = str(info.get('title', '')).lower()
            tokens = [t for t in re.split(r'\s+', title_text) if t]
            for token in tokens:
                if token not in title:
                    return False

        selected_submodel = submodel_combo.currentText()
        if selected_submodel != '(any)':
            expected_count = submodel_count.value()
            actual_count = int(info.get('submodel_counts', {}).get(selected_submodel, 0))
            if actual_count < expected_count:
                return False

        return True

    def fill_preview(info):
        if not info:
            info_label.setText('')
            preview_table.setRowCount(0)
            return

        md = info.get('metadata', {})
        comment_text = '\n'.join(info.get('comments', [])) if info.get('comments') else '(none)'
        active_models = info.get('active_models', [])
        model_sequence_text = '\n'.join(active_models) if active_models else '(none)'

        lines = [f"<b>Title:</b> {info.get('title', '')}"]
        for key in LIBRARY_METADATA_FIELDS:
            lines.append(f"<b>{key}:</b> {md.get(key, '')}")
        safe_comment = comment_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
        safe_sequence = model_sequence_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
        lines.append(f"<b>Comment:</b><br><i>{safe_comment}</i>")
        lines.append(f"<b>Model sequence:</b><br><i>{safe_sequence}</i>")

        warnings_list = info.get('validation_warnings', [])
        if warnings_list:
            safe_warnings = '<br>'.join(w.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') for w in warnings_list)
            lines.append(f"<b>Validation:</b><br><span style='color:#d58512'>{safe_warnings}</span>")

        info_label.setText('<br>'.join(lines))

        rows = info.get('table_rows', [])
        grouped = {}
        order = []
        for row in rows:
            idx = row.get('row_index', 0)
            if idx not in grouped:
                grouped[idx] = {'model': row.get('model', ''), 'params': []}
                order.append(idx)
            grouped[idx]['params'].append(row)

        # Result-table-like visualization: 3 rows per model
        preview_table.setRowCount(len(order) * 3)
        for g, idx in enumerate(order):
            group = grouped[idx]
            model_name = str(group.get('model', ''))
            params = group.get('params', [])
            base_row = g * 3

            for rr in [base_row, base_row + 1, base_row + 2]:
                text = model_name if rr == base_row else ''
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                preview_table.setItem(rr, 0, it)

            for c in range(1, numco + 1):
                for rr in [base_row, base_row + 1, base_row + 2]:
                    preview_table.setItem(rr, c, QTableWidgetItem(''))

            for p_i, p in enumerate(params[:numco]):
                c = p_i + 1
                p_name = str(p.get('param_name', ''))
                p_val = str(p.get('value', ''))
                p_lo = str(p.get('lower', ''))
                p_hi = str(p.get('upper', ''))
                p_fx = str(p.get('fixed', ''))

                preview_table.item(base_row, c).setText(p_name)
                preview_table.item(base_row + 1, c).setText(p_val)
                bound = f"[{p_lo}, {p_hi}]" if (p_lo or p_hi) else ''
                fix_text = 'fix' if p_fx.lower() == 'true' else ''
                preview_table.item(base_row + 2, c).setText(f"{bound} {fix_text}".strip())

        for r in range(preview_table.rowCount()):
            preview_table.setRowHeight(r, 22)

    def refresh_list():
        current_file = None
        current_item = list_widget.currentItem()
        if current_item is not None:
            current_file = current_item.data(Qt.ItemDataRole.UserRole)

        list_widget.clear()
        filtered = [info for info in parsed_items if item_matches_filters(info)]
        for info in filtered:
            item_text = info.get('list_text', info.get('file_name', ''))
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, info.get('file_name', ''))
            list_widget.addItem(item)

        if list_widget.count() == 0:
            fill_preview(None)
            return

        if current_file:
            for i in range(list_widget.count()):
                it = list_widget.item(i)
                if it.data(Qt.ItemDataRole.UserRole) == current_file:
                    list_widget.setCurrentRow(i)
                    break
            else:
                list_widget.setCurrentRow(0)
        else:
            list_widget.setCurrentRow(0)

    def on_list_selection_changed():
        current_item = list_widget.currentItem()
        if current_item is None:
            fill_preview(None)
            return
        file_name = current_item.data(Qt.ItemDataRole.UserRole)
        selected = next((i for i in parsed_items if i.get('file_name') == file_name), None)
        fill_preview(selected)

    list_widget.currentItemChanged.connect(lambda *_: on_list_selection_changed())
    submodel_combo.currentTextChanged.connect(
        lambda *_: (submodel_count.setEnabled(submodel_combo.currentText() != '(any)'), refresh_list())
    )
    submodel_count.valueChanged.connect(lambda *_: refresh_list())
    temp_from.textChanged.connect(lambda *_: refresh_list())
    temp_to.textChanged.connect(lambda *_: refresh_list())
    press_from.textChanged.connect(lambda *_: refresh_list())
    press_to.textChanged.connect(lambda *_: refresh_list())
    field_from.textChanged.connect(lambda *_: refresh_list())
    field_to.textChanged.connect(lambda *_: refresh_list())
    composition_filter.textChanged.connect(lambda *_: refresh_list())
    title_filter.textChanged.connect(lambda *_: refresh_list())

    refresh_list()

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    root_layout.addWidget(buttons)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        current_item = list_widget.currentItem()
        if current_item is None:
            QMessageBox.warning(parent_widget, "Library", "Please select a model file.")
            return
        selected_file = os.path.join(library_dir, current_item.data(Qt.ItemDataRole.UserRole))
        load_model_from_path(main_window, selected_file, insert_row=insert_row)
