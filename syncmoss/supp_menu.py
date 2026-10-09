"""
The "Supp" (support) button menu and its actions.

Everything reachable from the Supp button in the main window lives here:
the menu construction (:func:`build_supp_menu`), the light/dark theme toggle,
and the handlers for the small settings dialogs (integral points, instrumental
lines, polarization, the Be / KB impurity presets), the Library export/import,
the two markdown viewers
(models description and quick help), the License window, the (parked)
Hamiltonian initial-guess helper and the highlighted "Contact the author"
entry that closes the menu.

The values edited by the dialogs are stored in hidden QLineEdit widgets on
the main window (``jn0_input``, ``instrumental_number``,
``polarization_input``) because the fitting code reads them via
``.text()`` exactly like the visible inputs; only the editing UI moved
into dialogs.
"""

import os

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMenu, QMessageBox, QPushButton, QVBoxLayout,
)
from PySide6.QtGui import (
    QAction, QColor, QDoubleValidator, QIcon, QIntValidator, QPainter, QPen,
    QPixmap,
)
from PySide6.QtCore import QLocale, QPointF, QRectF, Qt

from syncmoss.constants import AUTHOR_EMAIL, ISSUES_URL
from syncmoss.error_reporter import app_version
from syncmoss.Library_io import export_library, import_library
from syncmoss.models_description_window import (
    ModelsDescriptionWindow, resolve_help_path, resolve_models_description_path,
)
from syncmoss.license_window import LicenseWindow
import syncmoss.sms_theory as smst
from syncmoss.instrumental_io import (
    DEFAULT_INSTRUMENTAL_METHOD, THEORY_BOUNDS, THEORY_START,
    get_sms_instrumental_from_global_files, read_accurate_instrumental,
    resolve_instrumental_for_file,
)
from syncmoss.instrumental_window import InstrumentalFunctionWindow
from syncmoss.parameters_table import DOUBLET_NAMES

# The two impurity presets of the model menu ('Be' and 'KB_nano'): a Doublet
# each, kept in params_dir as one tab-separated line of its nine values. They
# describe the CURRENT state of the beamline and change with it (data of
# another time may need other values), so they are editable here. The model
# menu, the calibration, the instrumental-function search and the results
# table (which reports a row equal to one as the impurity) all read the files.
IMPURITY_PRESETS = (
    ('Be.txt', "Be (optics impurity)"),
    ('KB.txt', "KB (Nanoscope impurity)"),
)

# Accent used to highlight the "Contact the author" entry, per mode. Qt offers
# no per-action text color (a QMenu stylesheet would repaint every item), so the
# entry is set apart by a bold font plus the envelope icon below — and the icon
# is the only colored part, which is why it gets one color per mode instead of
# a single "safe" one that would be washed out on one of the two backgrounds.
CONTACT_ACCENT_DARK = '#5aa9ff'
CONTACT_ACCENT_LIGHT = '#0b57d0'


def theme_action_text(is_dark_mode):
    """Label for the theme entry: it names the mode the click switches TO."""
    return "Switch to light mode" if is_dark_mode else "Switch to dark mode"


def contact_accent_color(is_dark_mode):
    """Accent color of the "Contact the author" entry for the active mode."""
    return CONTACT_ACCENT_DARK if is_dark_mode else CONTACT_ACCENT_LIGHT


def contact_icon(is_dark_mode):
    """A small envelope drawn in the accent color of the active mode.

    Painted rather than shipped as a file so it re-colors on every theme
    switch (``PhysicsApp._apply_theme`` re-sets it) and never sits invisibly
    on a background of its own color.
    """
    size = 32
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(contact_accent_color(is_dark_mode)))
        pen.setWidthF(2.6)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.drawRect(QRectF(3.5, 7.5, 25.0, 17.0))       # envelope body
        painter.drawLine(QPointF(3.5, 7.5), QPointF(16.0, 18.0))   # flap, left
        painter.drawLine(QPointF(28.5, 7.5), QPointF(16.0, 18.0))  # flap, right
    finally:
        painter.end()
    return QIcon(pixmap)


def build_supp_menu(main_window):
    """Create the Supp QMenu wired to *main_window* and return it.

    The theme entry is kept on the window as ``main_window.theme_action`` so
    ``PhysicsApp._apply_theme`` can relabel it whenever the mode changes; the
    contact entry is kept as ``main_window.contact_action`` for the same reason
    (its accent icon is repainted there).
    """
    menu = QMenu(main_window)
    menu.setToolTipsVisible(True)  # only the contact entry sets one

    theme_action = QAction(theme_action_text(main_window._is_dark_mode), main_window)
    theme_action.triggered.connect(main_window.toggle_theme)
    main_window.theme_action = theme_action

    ham_guess_action = QAction("Find initial guess for Hamiltonian", main_window)
    ham_guess_action.triggered.connect(lambda: open_hamiltonian_helper(main_window))
    set_integral_points_action = QAction("Set number of points for full transmission integral", main_window)
    set_integral_points_action.triggered.connect(lambda: open_integral_points_dialog(main_window))
    set_polarization_action = QAction("Set polarization", main_window)
    set_polarization_action.triggered.connect(lambda: open_polarization_dialog(main_window))
    ins_method_action = QAction(
        "Choose how to approximate instrumental function", main_window)
    ins_method_action.triggered.connect(
        lambda: open_instrumental_method_dialog(main_window))
    preset_actions = []
    for file_name, label in IMPURITY_PRESETS:
        action = QAction(f"Set parameters of {label}", main_window)
        action.triggered.connect(
            lambda _checked=False, f=file_name, l=label:
                open_impurity_preset_dialog(main_window, f, l))
        preset_actions.append(action)
    plot_ins_memory_action = QAction("Plot instrumental function from memory", main_window)
    plot_ins_memory_action.triggered.connect(lambda: plot_instrumental_from_memory(main_window))
    plot_ins_spectrum_action = QAction("Plot instrumental function from spectrum", main_window)
    plot_ins_spectrum_action.triggered.connect(lambda: plot_instrumental_from_spectrum(main_window))
    export_lib_action = QAction("Export Library", main_window)
    export_lib_action.triggered.connect(lambda: export_library_pressed(main_window))
    import_lib_action = QAction("Import Library", main_window)
    import_lib_action.triggered.connect(lambda: import_library_pressed(main_window))
    models_description_action = QAction("Models description", main_window)
    models_description_action.triggered.connect(lambda: open_models_description_pressed(main_window))
    help_action = QAction("Help (hidden features)", main_window)
    help_action.triggered.connect(lambda: open_help_pressed(main_window))
    license_action = QAction("License", main_window)
    license_action.triggered.connect(lambda: open_license_pressed(main_window))

    # Last entry, highlighted: bold + the accent envelope. Kept on the window as
    # ``main_window.contact_action`` so _apply_theme can re-color the icon.
    contact_action = QAction(contact_icon(main_window._is_dark_mode),
                             "Contact the author", main_window)
    contact_font = contact_action.font()
    contact_font.setBold(True)
    contact_action.setFont(contact_font)
    contact_action.setToolTip(f"Write to {AUTHOR_EMAIL}")
    contact_action.triggered.connect(lambda: contact_author_pressed(main_window))
    main_window.contact_action = contact_action

    menu.addAction(theme_action)
    menu.addSeparator()
    menu.addAction(ham_guess_action)
    menu.addAction(set_integral_points_action)
    # "Set number of lines to reconstruct the instrumental function" used to be
    # its own entry here. It is the number of GAUSSIANS, i.e. a property of one
    # of the two approximations, so it now lives beside that button inside the
    # method dialog. open_instrumental_lines_dialog() is kept for callers that
    # still want the standalone editor.
    menu.addAction(set_polarization_action)
    menu.addAction(ins_method_action)
    for action in preset_actions:
        menu.addAction(action)
    menu.addSeparator()
    menu.addAction(plot_ins_memory_action)
    menu.addAction(plot_ins_spectrum_action)
    menu.addSeparator()
    menu.addAction(export_lib_action)
    menu.addAction(import_lib_action)
    menu.addAction(models_description_action)
    menu.addAction(help_action)
    menu.addAction(license_action)
    menu.addSeparator()
    menu.addAction(contact_action)
    return menu


def _wrapped_label(text, parent):
    """A word-wrapping QLabel — the dialog below is mostly these."""
    label = QLabel(text, parent)
    label.setWordWrap(True)
    return label


class ContactDialog(QDialog):
    """Small window with the author's address and the issue-tracker link.

    Deliberately does NOT open a ``mailto:`` URL. That hands the message to
    whatever mail client the machine has registered — Outlook, typically —
    which is rarely the one the user actually writes from, and on a machine
    with none registered it does nothing visible at all. The address is simply
    shown, selectable, with a button that copies it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Contact the author")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)

        layout.addWidget(_wrapped_label(
            "Questions, bug reports and feature requests are all welcome.", self))

        self.email_label = QLabel(AUTHOR_EMAIL, self)
        email_font = self.email_label.font()
        email_font.setBold(True)
        email_font.setPointSize(email_font.pointSize() + 2)
        self.email_label.setFont(email_font)
        self.email_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        layout.addWidget(self.email_label)

        issues = QLabel(f'or open an issue: <a href="{ISSUES_URL}">{ISSUES_URL}</a>', self)
        issues.setTextFormat(Qt.TextFormat.RichText)
        issues.setOpenExternalLinks(True)   # a browser, not a mail client
        issues.setWordWrap(True)
        layout.addWidget(issues)

        layout.addWidget(_wrapped_label(
            "For a bug report please attach the spectrum and the model (.mdl) "
            "file, and say which button or action caused the problem.", self))

        version = app_version()
        if version:
            layout.addWidget(_wrapped_label(f"This is SYNCmoss {version}.", self))

        self.hint = QLabel("", self)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        buttons = QHBoxLayout()
        self.copy_btn = QPushButton("Copy the e-mail address", self)
        self.copy_btn.clicked.connect(self.copy_email)
        close_btn = QPushButton("Close", self)
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(self.copy_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def copy_email(self):
        clipboard = QApplication.clipboard()
        if clipboard is None:
            self.hint.setText("No clipboard available - please select the address and copy it.")
            return False
        clipboard.setText(AUTHOR_EMAIL)
        self.hint.setText("Address copied to the clipboard.")
        return True


def contact_author_pressed(main_window):
    """Show the author's address and the issue tracker."""
    main_window.set_status(f"Write to {AUTHOR_EMAIL}", "blue")
    ContactDialog(main_window).exec()


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


INTEGRAL_POINTS_MIN = 1
INTEGRAL_POINTS_MAX = 1000000


def open_integral_points_dialog(main_window):
    """Edit the transmission-integral point counts, one per source type.

    SMS and CMS need different numbers, so they are two independent settings
    (defaults 32 and 64). There used to be a switch instead: entering CMS
    doubled 32 to 64 and leaving it halved back, but only while the value was
    still exactly one of those two -- so any edited value silently stopped
    tracking the mode. Both are simply settable now, and
    ``refresh_jn0_for_mode`` selects the active one.
    """
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Set number of points for full transmission integral")
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Number of points for the full transmission "
                            "integral, per source type:"))

    editors = {}
    for key, label, store in (
            ('sms', "SMS:", main_window.jn0_sms_input),
            ('cms', "CMS:", main_window.jn0_cms_input)):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        editor = QLineEdit(store.text().strip(), dialog)
        editor.setValidator(QIntValidator(INTEGRAL_POINTS_MIN,
                                          INTEGRAL_POINTS_MAX, dialog))
        editor.setMaximumWidth(110)
        row.addWidget(editor)
        row.addStretch(1)
        layout.addLayout(row)
        editors[key] = (editor, store)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(420)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False

    # validate BOTH before storing either, so a typo in one does not leave the
    # pair half-applied
    values = {}
    for key, (editor, _store) in editors.items():
        text = editor.text().strip()
        try:
            value = int(text)
        except ValueError:
            QMessageBox.warning(main_window, "Integral points",
                                f"{key.upper()}: please enter a whole number.")
            return False
        if not INTEGRAL_POINTS_MIN <= value <= INTEGRAL_POINTS_MAX:
            QMessageBox.warning(
                main_window, "Integral points",
                f"{key.upper()}: value must be between {INTEGRAL_POINTS_MIN} "
                f"and {INTEGRAL_POINTS_MAX}.")
            return False
        values[key] = value

    for key, (_editor, store) in editors.items():
        store.setText(str(values[key]))
    main_window.refresh_jn0_for_mode()
    main_window.set_status(
        f"Integral points set to: SMS {values['sms']}, CMS {values['cms']}",
        "blue")
    return True


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


INSTRUMENTAL_METHODS = (
    ('theory', "Theory",
     "Simulation of 57FeBO3 source.\n"
     "A few PHYSICAL numbers instad of empirical ones\n"
     "It keeps the proper tails that.\n"
     "Technicaly a proper way but slower calculations"),
    ('gauss', "Set of Gaussians (default)",
     "The empirical description: a free sum of Gaussians. Faster and\n"
     "always able to follow the data, but the parameters mean nothing\n"
     "physical and a Gaussian sum has short tails.\n"
     "So far no issue was found by using it."),
)


def open_instrumental_method_dialog(main_window):
    """Choose how "Find / Refine Instr. func." approximates the source.

    This picks the SEARCH method only. A spectrum that carries its own
    instrumental function in its .dat file is unaffected -- what happens to
    those is governed by the separate "use instrumental function from .dat
    file" toggle on the Instrumental function button.

    The number of Gaussians belongs to the Gaussian option, so it is edited
    here beside that button rather than from its own Supp entry. It is written
    back to ``main_window.instrumental_number`` (the hidden store the fitting
    code reads) whichever button is pressed, so the count can be corrected
    without switching method.
    """
    current = getattr(main_window, 'instrumental_method',
                      DEFAULT_INSTRUMENTAL_METHOD)
    dialog = QDialog(main_window)
    dialog.setWindowTitle("How to approximate the instrumental function")
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(
        "How should \"Find\" and \"Refine Instr. func.\" describe the source?"))

    chosen = {'key': current}
    n_editor = QLineEdit(main_window.instrumental_number.text().strip(), dialog)
    n_editor.setValidator(QIntValidator(1, 1000, dialog))
    n_editor.setMaximumWidth(70)

    def pick(key):
        chosen['key'] = key
        dialog.accept()

    for key, title, blurb in INSTRUMENTAL_METHODS:
        mark = "  ✓" if key == current else ""
        button = QPushButton(f"{title}{mark}")
        button.setMinimumHeight(34)
        button.setToolTip(blurb)
        button.clicked.connect(lambda _checked=False, k=key: pick(k))
        if key == 'gauss':
            row = QHBoxLayout()
            row.addWidget(button, 1)
            row.addWidget(QLabel("number of Gaussians:"))
            row.addWidget(n_editor)
            layout.addLayout(row)
        else:
            layout.addWidget(button)
        note = QLabel(blurb)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

    layout.addWidget(QLabel(
        "Spectra that carry their own instrumental function in the .dat file\n"
        "are not affected by this setting."))
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(520)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return current

    # the count is stored whichever button was pressed -- it is a property of
    # the Gaussian description, not of the act of selecting it
    n_text = n_editor.text().strip()
    if n_text and n_text.isdigit() and 1 <= int(n_text) <= 1000:
        main_window.instrumental_number.setText(str(int(n_text)))
    elif n_text:
        QMessageBox.warning(main_window, "Number of Gaussians",
                            "Please enter a whole number between 1 and 1000; "
                            "the previous value was kept.")

    main_window.instrumental_method = chosen['key']
    label = dict((k, t) for k, t, _b in INSTRUMENTAL_METHODS)[chosen['key']]
    extra = (f" ({main_window.instrumental_number.text()} Gaussians)"
             if chosen['key'] == 'gauss' else "")
    main_window.set_status(f"Instrumental function: {label}{extra}", "green")
    print(f"[Supp] instrumental function approximated by: {label}{extra}")
    return chosen['key']


def open_theory_find_dialog(main_window):
    """Confirm and seed a THEORY "Find", or return None to cancel it.

    Find throws away whatever is stored and starts from built-in values, so it
    is the wrong button most of the time -- Refine continues from a shape
    already fitted to this source and is both faster and safer. The dialog says
    so, and then asks for the four numbers the user may actually know better
    than the defaults do.

    B_s and shift are deliberately NOT asked: Find MAPS both of them over a
    grid (THEORY_BS_SCAN x THEORY_SHIFT_SCAN) before either is released. The
    velocity zero is the number a user is least able to supply -- it belongs to
    the drive calibration of one measurement -- so asking for it would be
    asking for the thing the map is there to find.

    Returns a dict of overrides for THEORY_START, or None if cancelled.
    """
    fields = (
        ('theta_urad', "Rocking angle θ [µrad]",
         "The position you set on the rocking curve. Held during the search --\n"
         "released from one spectrum it is invented, not measured."),
        ('dEQ', "Quadrupole splitting ΔE_Q [mm/s]",
         "A property of FeBO3. Held; -0.3900 comes from the 83-spectrum fit."),
        ('f_LM', "Nuclear amplitude scale f_LM",
         "f_LM x enrichment of the source crystal. Held; 0.7718 is both the\n"
         "Debye value and where the 83-spectrum fit puts it."),
    )
    start = dict(THEORY_START)

    dialog = QDialog(main_window)
    dialog.setWindowTitle("Find the instrumental function (Theory)")
    layout = QVBoxLayout(dialog)
    warn = QLabel(
        "<b>“Refine” is usually the better choice.</b><br>"
        "Find discards the stored instrumental function and restarts from the "
        "built-in values.<br>Refine continues from the shape already fitted to "
        "this source — faster, and it cannot wander as far.")
    warn.setWordWrap(True)
    layout.addWidget(warn)
    layout.addWidget(QLabel(
        "\nStarting values — change them if this source was run far from "
        "the usual settings.\nB_s and the velocity shift are not asked: Find "
        "maps both of them."))

    editors = {}
    for key, label, blurb in fields:
        row = QHBoxLayout()
        lab = QLabel(label)
        lab.setMinimumWidth(230)
        row.addWidget(lab)
        editor = QLineEdit(f"{float(start[key]):g}", dialog)
        validator = QDoubleValidator(-1e6, 1e6, 6, dialog)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale(QLocale.Language.C))
        editor.setValidator(validator)
        editor.setMaximumWidth(120)
        row.addWidget(editor)
        row.addStretch(1)
        layout.addLayout(row)
        note = QLabel(blurb)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
        editors[key] = editor

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(560)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    # validate ALL of them before returning any, and against the same bounds
    # the search itself uses -- a start outside them is refused there anyway
    out = {}
    bounds = THEORY_BOUNDS
    for key, label, _blurb in fields:
        text = editors[key].text().strip().replace(',', '.')
        try:
            value = float(text)
        except ValueError:
            QMessageBox.warning(main_window, "Find instrumental function",
                                f"{label}: please enter a number.")
            return None
        lo, hi = bounds[key]
        if not lo <= value <= hi:
            QMessageBox.warning(
                main_window, "Find instrumental function",
                f"{label}: must be between {lo:g} and {hi:g}.")
            return None
        out[key] = value
    return out


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


def open_impurity_preset_dialog(main_window, file_name, label):
    """Edit an impurity preset: the nine Doublet values in params_dir/*file_name*.

    The fields show the file's own texts, so OK without a change writes the same
    numbers back. On OK every field must hold a number; the file is then
    rewritten as one tab-separated line. Rows already in the table keep their
    numbers -- picking the preset in a row again loads the new ones. Returns
    True when the file was written.
    """
    path = os.path.join(main_window.params_dir, file_name)
    try:
        with open(path, encoding='utf-8') as f:
            current = f.read().split()
    except OSError:
        current = []
    unreadable = len(current) != len(DOUBLET_NAMES)
    if unreadable:
        current = [''] * len(DOUBLET_NAMES)

    title = f"Set parameters of {label}"
    dialog = QDialog(main_window)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(_wrapped_label(
        f"The Doublet the model menu puts in a row for {label} -- the current "
        f"state of the beamline, kept in {file_name}. Rows already in the table "
        f"keep their numbers; pick it again in a row to load the new ones.", dialog))
    if unreadable:
        layout.addWidget(_wrapped_label(
            f"{file_name} could not be read: enter all {len(DOUBLET_NAMES)} values.", dialog))

    editors = []
    for name, text in zip(DOUBLET_NAMES, current):
        row = QHBoxLayout()
        name_label = QLabel(name)
        name_label.setMinimumWidth(90)
        row.addWidget(name_label)
        editor = QLineEdit(text, dialog)
        validator = QDoubleValidator(-1e9, 1e9, 10, dialog)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale(QLocale.Language.C))
        editor.setValidator(validator)
        editor.setMaximumWidth(140)
        row.addWidget(editor)
        row.addStretch(1)
        layout.addLayout(row)
        editors.append(editor)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(420)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False

    # validate all nine before writing any of them
    texts = []
    for name, editor in zip(DOUBLET_NAMES, editors):
        text = editor.text().strip()
        try:
            valid = bool(np.isfinite(float(text)))
        except ValueError:
            valid = False
        if not valid:
            QMessageBox.warning(main_window, title,
                                f"{name}: please enter a number; {file_name} was not changed.")
            return False
        texts.append(text)

    try:
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\t'.join(texts) + '\n')
    except OSError as e:
        QMessageBox.warning(main_window, title, f"Could not write {path}:\n{e}")
        main_window.set_status(f"Could not save the parameters of {label}: {e}", "red")
        return False
    main_window.set_status(f"Parameters of {label} saved to {path}", "blue")
    return True


def _draw_instrumental(main_window, curves, title, note):
    """Show instrumental-function curves in their OWN window and report ``note``.

    Deliberately not the main canvas: the instrumental function is a property of
    the source, not a result of the current fit, and the user wants it visible
    while working on the spectrum. One window is created on first use and reused.
    """
    try:
        window = getattr(main_window, 'instrumental_window', None)
        if window is None:
            window = InstrumentalFunctionWindow(main_window)
            main_window.instrumental_window = window
        icon = main_window.windowIcon()
        if not icon.isNull():
            window.setWindowIcon(icon)
        saved = window.show_curves(
            curves, title, note, main_window.dir_path,
            theme=main_window._theme, gridcolor=main_window.gridcolor)
        main_window.set_status(
            note + (f"\nSaved to {saved[1]}" if saved else ""), "green")
    except Exception as e:
        import traceback
        main_window.set_status(
            f"Could not plot the instrumental function: {e}\n{traceback.format_exc()}",
            "red")


def _instrumental_curves(INS, v_max=3.0, n=6001):
    """(label, v, S, color) for an instrumental function, plus a one-line summary.

    ONE curve: whichever description ``INS`` actually is. A theoretical
    instrumental function used to be drawn together with a Gaussian-sum
    stand-in for the same source, but the two descriptions are now a setting
    (Supp -> "Choose how to approximate instrumental function") and the plot
    shows the one in use -- drawing the other alongside it invites reading the
    stand-in as if it were what the program is using.
    """
    v = np.linspace(-v_max, v_max, n)
    kind = smst.ins_kind(INS)
    S = smst.ins_shape(INS, v)
    fwhm, centre, _area = smst.line_metrics(v, S)
    summary = (f"{smst.describe_ins(INS)}\n"
               f"FWHM {fwhm:.4f} mm/s = {fwhm / smst.NAT_WIDTH:.2f} natural widths, "
               f"centre {centre:+.4f} mm/s, first moment "
               f"{smst.ins_centroid(INS):+.4f} mm/s")
    label = ("Simulation of 57FeBO3" if kind != smst.KIND_GAUSS
             else "empirical sum of Gaussians")
    return [(label, v, S, 'red')], summary


def plot_instrumental_from_memory(main_window):
    """Plot the instrumental function the program is currently using.

    Which one that is follows the setting in "Choose how to approximate
    instrumental function": the simulated shape from INSth.txt, or the Gaussian
    sum from INSexp.txt. It used to show whichever file happened to exist, so
    selecting the Gaussians still plotted the theoretical shape.
    """
    if not main_window.initialize_parameters():
        return
    try:
        if main_window.MS_fit.isChecked():
            main_window.set_status(
                "CMS mode: the instrumental function is the single Gaussian width G "
                f"= {main_window.GCMS}, there is nothing to plot.", "orange")
            return
        INS, MulCo, x0 = get_sms_instrumental_from_global_files(main_window)
        # name what was ACTUALLY returned, not what is on disk: the setting can
        # select a description that has not been found yet, and the loader then
        # falls back to the other one
        source = ("INSth.txt (simulated 57FeBO3)"
                  if smst.ins_kind(INS) != smst.KIND_GAUSS
                  else "INSexp.txt (sum of Gaussians)")
    except Exception as e:
        main_window.set_status(f"Could not read the instrumental function: {e}", "red")
        return

    curves, summary = _instrumental_curves(INS)
    title = f"Instrumental function in memory — {source}"
    _draw_instrumental(main_window, curves, title,
                       f"{title}\n{summary}\nMulCo = {MulCo:.5f}, x0 = {x0:.5f}")


def plot_instrumental_from_spectrum(main_window):
    """Plot the instrumental function written into the loaded spectrum file.

    Resolved exactly the way a fit of that spectrum would resolve it (its
    #@INSth or #@INSexp with #@INSint, else the internal values), so what is
    drawn is what the fit would actually use.
    """
    if not main_window.initialize_parameters():
        return
    if not main_window.path_list:
        main_window.set_status("No spectrum loaded", "red")
        return
    spectrum_file = os.path.abspath(main_window.path_list[0])
    use_dat = bool(getattr(main_window, 'use_dat_instrumental_metadata', True))
    try:
        mp = resolve_instrumental_for_file(main_window, spectrum_file,
                                           use_dat_metadata=use_dat)
    except Exception as e:
        main_window.set_status(f"Could not resolve the instrumental function: {e}", "red")
        return

    if mp['method'] == 'CMS':
        main_window.set_status(
            f"{os.path.basename(spectrum_file)} is a CMS spectrum: its instrumental "
            f"function is the single Gaussian width G = {mp['INS']}, "
            f"there is nothing to plot.\n{mp['note']}", "orange")
        return

    curves, summary = _instrumental_curves(np.atleast_1d(np.asarray(mp['INS'], float)))
    title = f"Instrumental function of {os.path.basename(spectrum_file)}"
    _draw_instrumental(
        main_window, curves, title,
        f"{title}\n{mp['note']}\n{summary}\n"
        f"MulCo = {mp['MulCo']:.5f}, x0 = {mp['x0']:.5f}")


def open_hamiltonian_helper(main_window):
    """Open placeholder support widget for Hamiltonian initial guess."""
    main_window.set_status("Not yet implemented: Hamiltonian helper coming in future update", "orange")
    # Parked feature — enable once the widget is functional:
    # from syncmoss.Hamiltonian_helper import HamiltonianHelperWidget
    # main_window._ham_helper_window = HamiltonianHelperWidget(main_window)
    # main_window._ham_helper_window.exec()


def export_library_pressed(main_window):
    """Export internal Library folder to a selected destination."""
    # The Library dialogs open in the Library folder itself, not the work folder.
    destination = QFileDialog.getExistingDirectory(
        main_window, "Select destination folder", main_window.library_dir)
    if not destination:
        main_window.set_status("Export Library canceled", "orange")
        return
    try:
        library_dir = main_window.library_dir
        target = export_library(library_dir, destination)
        main_window.set_status(f"Library exported to: {target}", "green")
    except Exception as e:
        main_window.set_status(f"Export Library failed: {e}", "red")


def import_library_pressed(main_window):
    """Import .mdl files from selected folder into internal Library folder."""
    source = QFileDialog.getExistingDirectory(
        main_window, "Select source folder", main_window.library_dir)
    if not source:
        main_window.set_status("Import Library canceled", "orange")
        return
    try:
        library_dir = main_window.library_dir
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
    _bring_to_front(main_window, window)


def _bring_to_front(main_window, window):
    """Show a helper window with the app icon, above the main window."""
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


def open_license_pressed(main_window):
    """The license of SYNCmoss and of the third-party components it ships with."""
    window = getattr(main_window, 'license_window', None)
    if window is None:
        window = LicenseWindow(main_window.dir_path)
        main_window.license_window = window
    else:
        window.reload()
    _bring_to_front(main_window, window)
