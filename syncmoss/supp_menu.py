"""
The "Supp" (support) button menu and its actions.

Everything reachable from the Supp button in the main window lives here:
the menu construction (:func:`build_supp_menu`), the light/dark theme toggle,
and the handlers for the small settings dialogs (integral points, instrumental
lines, polarization), the Library export/import, the two markdown viewers
(models description and quick help) and the (parked) Hamiltonian initial-guess
helper.

The values edited by the dialogs are stored in hidden QLineEdit widgets on
the main window (``jn0_input``, ``instrumental_number``,
``polarization_input``) because the fitting code reads them via
``.text()`` exactly like the visible inputs; only the editing UI moved
into dialogs.
"""

import os

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QLabel, QLineEdit, QMenu,
    QMessageBox, QVBoxLayout,
)
from PySide6.QtGui import QAction, QDoubleValidator, QIntValidator
from PySide6.QtCore import QLocale

from syncmoss.Library_io import export_library, import_library
from syncmoss.models_description_window import (
    ModelsDescriptionWindow, resolve_help_path, resolve_models_description_path,
)


def theme_action_text(is_dark_mode):
    """Label for the theme entry: it names the mode the click switches TO."""
    return "Switch to light mode" if is_dark_mode else "Switch to dark mode"


def build_supp_menu(main_window):
    """Create the Supp QMenu wired to *main_window* and return it.

    The theme entry is kept on the window as ``main_window.theme_action`` so
    ``PhysicsApp._apply_theme`` can relabel it whenever the mode changes.
    """
    menu = QMenu(main_window)

    theme_action = QAction(theme_action_text(main_window._is_dark_mode), main_window)
    theme_action.triggered.connect(main_window.toggle_theme)
    main_window.theme_action = theme_action

    ham_guess_action = QAction("Find initial guess for Hamiltonian", main_window)
    ham_guess_action.triggered.connect(lambda: open_hamiltonian_helper(main_window))
    set_integral_points_action = QAction("Set number of points for full transmission integral", main_window)
    set_integral_points_action.triggered.connect(lambda: open_integral_points_dialog(main_window))
    set_instrumental_lines_action = QAction("Set number of lines to reconstruct the instrumental function", main_window)
    set_instrumental_lines_action.triggered.connect(lambda: open_instrumental_lines_dialog(main_window))
    set_polarization_action = QAction("Set polarization", main_window)
    set_polarization_action.triggered.connect(lambda: open_polarization_dialog(main_window))
    export_lib_action = QAction("Export Library", main_window)
    export_lib_action.triggered.connect(lambda: export_library_pressed(main_window))
    import_lib_action = QAction("Import Library", main_window)
    import_lib_action.triggered.connect(lambda: import_library_pressed(main_window))
    models_description_action = QAction("Models description", main_window)
    models_description_action.triggered.connect(lambda: open_models_description_pressed(main_window))
    help_action = QAction("Help (hidden features)", main_window)
    help_action.triggered.connect(lambda: open_help_pressed(main_window))

    menu.addAction(theme_action)
    menu.addSeparator()
    menu.addAction(ham_guess_action)
    menu.addAction(set_integral_points_action)
    menu.addAction(set_instrumental_lines_action)
    menu.addAction(set_polarization_action)
    menu.addSeparator()
    menu.addAction(export_lib_action)
    menu.addAction(import_lib_action)
    menu.addAction(models_description_action)
    menu.addAction(help_action)
    return menu


def _open_value_setting_dialog(main_window, title, label_text, target_input,
                               min_value, max_value, parse=int, decimals=6):
    """Open a compact modal dialog to edit one numeric setting.

    ``parse`` selects integer or float mode (validator + parsing + the range
    check). On accept the validated value is written back into
    ``target_input`` (one of the hidden setting stores) and True is returned;
    any cancel/validation failure returns False and leaves the store as-is.
    """
    dialog = QDialog(main_window)
    dialog.setWindowTitle(title)

    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(label_text))

    editor = QLineEdit(target_input.text().strip(), dialog)
    if parse is int:
        editor.setValidator(QIntValidator(min_value, max_value, dialog))
    else:
        validator = QDoubleValidator(min_value, max_value, decimals, dialog)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale(QLocale.Language.C))
        editor.setValidator(validator)
    editor.selectAll()
    layout.addWidget(editor)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    dialog.setMinimumWidth(460)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False

    value_text = editor.text().strip()
    if not value_text:
        QMessageBox.warning(main_window, title, "Value cannot be empty.")
        return False

    try:
        value = parse(value_text)
    except ValueError:
        kind = "integer" if parse is int else "number"
        QMessageBox.warning(main_window, title, f"Please enter a valid {kind}.")
        return False

    if value < min_value or value > max_value:
        QMessageBox.warning(main_window, title, f"Value must be between {min_value} and {max_value}.")
        return False

    target_input.setText(str(value))
    return True


def open_integral_points_dialog(main_window):
    """Edit the number of points used for full transmission integral."""
    changed = _open_value_setting_dialog(
        main_window,
        "Set number of points for full transmission integral",
        "Number of points for full transmission integral:",
        main_window.jn0_input,
        min_value=1,
        max_value=1000000,
        parse=int,
    )
    if changed:
        main_window.set_status(f"Integral points set to: {main_window.jn0_input.text()}", "blue")


def open_instrumental_lines_dialog(main_window):
    """Edit the number of lines used to reconstruct the instrumental function."""
    changed = _open_value_setting_dialog(
        main_window,
        "Set number of lines to reconstruct the instrumental function",
        "Number of lines to reconstruct the instrumental function:",
        main_window.instrumental_number,
        min_value=1,
        max_value=1000,
        parse=int,
    )
    if changed:
        main_window.set_status(
            f"Number of lines for instrumental function set to: {main_window.instrumental_number.text()}", "blue")


def open_polarization_dialog(main_window):
    """Edit the SMS beam linear polarization degree (0..1)."""
    changed = _open_value_setting_dialog(
        main_window,
        "Set polarization",
        "SMS linear polarization degree (0 = unpolarized, 1 = fully polarized):",
        main_window.polarization_input,
        min_value=0.0,
        max_value=1.0,
        parse=float,
    )
    if changed:
        main_window.SMS_pol = float(main_window.polarization_input.text())
        main_window.set_status(f"Polarization set to: {main_window.polarization_input.text()}", "blue")


def open_hamiltonian_helper(main_window):
    """Open placeholder support widget for Hamiltonian initial guess."""
    main_window.set_status("Not yet implemented: Hamiltonian helper coming in future update", "orange")
    # Parked feature — enable once the widget is functional:
    # from syncmoss.Hamiltonian_helper import HamiltonianHelperWidget
    # main_window._ham_helper_window = HamiltonianHelperWidget(main_window)
    # main_window._ham_helper_window.exec()


def export_library_pressed(main_window):
    """Export internal Library folder to a selected destination."""
    destination = QFileDialog.getExistingDirectory(
        main_window, "Select destination folder", main_window.workfolder or main_window.dir_path)
    if not destination:
        main_window.set_status("Export Library canceled", "orange")
        return
    try:
        library_dir = os.path.join(main_window.dir_path, 'Library')
        target = export_library(library_dir, destination)
        main_window.set_status(f"Library exported to: {target}", "green")
    except Exception as e:
        main_window.set_status(f"Export Library failed: {e}", "red")


def import_library_pressed(main_window):
    """Import .mdl files from selected folder into internal Library folder."""
    source = QFileDialog.getExistingDirectory(
        main_window, "Select source folder", main_window.workfolder or main_window.dir_path)
    if not source:
        main_window.set_status("Import Library canceled", "orange")
        return
    try:
        library_dir = os.path.join(main_window.dir_path, 'Library')
        result = import_library(source, library_dir)
        if isinstance(result, dict):
            copied = int(result.get('copied', 0))
            skipped_identical = int(result.get('skipped_identical', 0))
            renamed = list(result.get('renamed', []))
        else:
            # Backward compatibility fallback
            copied = int(result)
            skipped_identical = 0
            renamed = []

        main_window.set_status(
            f"Imported {copied} .mdl file(s) into Library"
            + (f" (skipped identical: {skipped_identical})" if skipped_identical else ""),
            "green",
        )

        if copied == 0:
            QMessageBox.information(
                main_window,
                "Import Library",
                "Nothing was added: all imported models already exist in Library."
            )

        if renamed:
            lines = [f"{old} -> {new}" for old, new in renamed]
            QMessageBox.information(
                main_window,
                "Library versions created",
                "Some imported models matched existing titles and were saved as new versions:\n\n"
                + "\n".join(lines)
            )
    except Exception as e:
        main_window.set_status(f"Import Library failed: {e}", "red")


def _open_markdown_document(main_window, doc_path, attribute, label):
    """Show *doc_path* in the markdown viewer kept on ``main_window.<attribute>``.

    One viewer window per document (created on first use, reloaded from disk on
    every later call) so the models description and the quick help can be open
    side by side. ``label`` is both the window title suffix and the wording used
    in the not-found messages.
    """
    if not os.path.isfile(doc_path):
        main_window.set_status(f"{label} file not found: {doc_path}", "red")
        QMessageBox.warning(main_window, label, f"File not found:\n{doc_path}")
        return

    window = getattr(main_window, attribute, None)
    if window is None:
        window = ModelsDescriptionWindow(doc_path, title=f"SYNCmoss - {label}")
        setattr(main_window, attribute, window)
    else:
        window.markdown_path = doc_path
        window.reload_document()

    app_icon = main_window.windowIcon()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)

    window.showNormal()
    window.show()
    window.raise_()
    window.activateWindow()


def open_models_description_pressed(main_window):
    """Open model descriptions markdown in a separate, copy-friendly window."""
    _open_markdown_document(main_window,
                            resolve_models_description_path(main_window.dir_path),
                            'models_description_window', "Models description")


def open_help_pressed(main_window):
    """Open the quick-help markdown (features not visible in the UI)."""
    _open_markdown_document(main_window,
                            resolve_help_path(main_window.dir_path),
                            'help_window', "Help")
