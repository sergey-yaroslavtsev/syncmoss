"""The dialogs the Instrumental function button's menu opens.

* the theory Find's start values (open_theory_find_dialog);
* the source's numbers of "Refine Δθ: pure α-Fe" (open_dtheta_alpha_fe_dialog);
* the start values and each spectrum's counts of "Calibrate + find G, Δθ: α-Fe
  + single line" (open_source_calibration_dialog,
  CMS_source_geometry_calibration).

Each returns the numbers it asked for, or None when cancelled or when a number
is refused (after saying why). The Supp button's own dialogs are in supp_menu.
"""
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout,
)
from PySide6.QtGui import QDoubleValidator
from PySide6.QtCore import QLocale

from syncmoss.CMS_source_geometry_calibration import start_values
from syncmoss.instrumental_io import THEORY_BOUNDS, THEORY_START, get_internal_dtheta, get_internal_gcms
from syncmoss.parameters_table import CMS_SOURCE_DECIMALS, ValueLineEdit


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
         "Your rocking-curve setting; held."),
        ('dEQ', "Quadrupole splitting ΔE_Q [mm/s]",
         "A property of FeBO3; held (-0.3900: the 83-spectrum fit)."),
        ('f_LM', "Nuclear amplitude scale f_LM",
         "f_LM x enrichment of the source crystal; held (0.7718: Debye value)."),
    )
    start = dict(THEORY_START)

    dialog = QDialog(main_window)
    dialog.setWindowTitle("Find the instrumental function (Theory)")
    layout = QVBoxLayout(dialog)
    warn = QLabel(
        "<b>“Refine” is usually the better choice:</b> Find restarts from the "
        "built-in values.")
    warn.setWordWrap(True)
    layout.addWidget(warn)
    layout.addWidget(QLabel(
        "\nStart values (B_s and the velocity shift are mapped by Find):"))

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


# The source's f_LM the Δθ dialogs propose.
DTHETA_SOURCE_F_LM = 0.7


def open_dtheta_alpha_fe_dialog(main_window):
    """The source's numbers for "Refine Δθ: pure α-Fe", or None to cancel it.

    G is held while Δθ is fitted (prefilled from the G field); f_LM, C_high and
    the static noise set the non-resonant baseline
    (instrumental_io.dtheta_alpha_fe_reference). The dialog says how to find
    them first.

    Returns {'G', 'f_LM', 'C_high', 'C_static'}, or None if cancelled.
    """
    fields = (
        ('G', "G [mm/s]", str(get_internal_gcms(main_window)),
         lambda v: v >= 0, "must not be negative"),
        ('f_LM', "f_LM of the source", str(DTHETA_SOURCE_F_LM),
         lambda v: 0 < v < 1, "must be between 0 and 1"),
        ('C_high', "C_high [counts/point]", "0",
         lambda v: v >= 0, "must not be negative"),
        ('C_static', "Static noise [counts/point]", "0",
         lambda v: v >= 0, "must not be negative"),
    )
    return _numbers_dialog(main_window, "Refine Δθ: pure α-Fe", (
        "Nnr = Ns (1 − f_LM)/f_LM + C_high; static noise added flat.",
        "<b>First:</b> G and f_LM from a single-line spectrum. C_high: count rate with a "
        "100 µm Cu foil × time per point, minus the static noise."), fields)


def open_source_calibration_dialog(main_window):
    """The numbers of "Calibrate + find G, Δθ: α-Fe + single line"
    (CMS_source_geometry_calibration.calibrate_source), or None to cancel it:
    where G, Δθ and f_LM start (G and Δθ from their fields, or the usual source
    when they hold none), and each spectrum's C_high and static noise per
    channel, which set its non-resonant baseline -- one pair per spectrum, as
    their counting times differ.

    Returns {'G', 'dtheta', 'f_LM', 'C_high': (α-Fe, single line),
    'C_static': (α-Fe, single line)}, or None if cancelled.
    """
    g, dtheta = start_values(get_internal_gcms(main_window), get_internal_dtheta(main_window))

    def counts_field(key, label):
        return key, label, "0", lambda v: v >= 0, "must not be negative"
    fields = (
        ('G', "G, start [mm/s]", str(g), lambda v: 0 < v < 1, "must be between 0 and 1"),
        ('dtheta', "Δθ, start [rad]", str(dtheta), lambda v: 0 < v < np.pi / 2, "must be between 0 and π/2"),
        ('f_LM', "f_LM of the source, start", str(DTHETA_SOURCE_F_LM),
         lambda v: 0 < v < 1, "must be between 0 and 1"),
        counts_field('C_high_alpha', "α-Fe: C_high [counts/channel]"),
        counts_field('C_static_alpha', "α-Fe: static noise [counts/channel]"),
        counts_field('C_high_line', "Single line: C_high [counts/channel]"),
        counts_field('C_static_line', "Single line: static noise [counts/channel]"),
    )
    out = _numbers_dialog(main_window, "Calibrate + find G, Δθ", (
        "Path box: α-Fe, then the single line (raw, one drive). Fitted together: the drive, G, Δθ, "
        "f_LM, and each spectrum's T and count rate.",
        "Nnr = Ns (1 − f_LM)/f_LM + C_high; static noise added flat. C_high: count rate with a "
        "100 µm Cu foil × time per channel, minus the static noise."), fields)
    if out is None:
        return None
    return {'G': out['G'], 'dtheta': out['dtheta'], 'f_LM': out['f_LM'],
            'C_high': (out['C_high_alpha'], out['C_high_line']),
            'C_static': (out['C_static_alpha'], out['C_static_line'])}


def _numbers_dialog(main_window, title, notes, fields):
    """A dialog of number fields under *notes*; *fields* are (key, label,
    text, check, why). Returns {key: value}, or None when cancelled or when a
    value is refused (after saying why)."""
    dialog = QDialog(main_window)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    for note in notes:
        label = QLabel(note, dialog)
        label.setWordWrap(True)
        layout.addWidget(label)

    editors = {}
    for key, label, text, _check, _why in fields:
        row = QHBoxLayout()
        lab = QLabel(label)
        lab.setMinimumWidth(200)
        row.addWidget(lab)
        # every digit kept, as in the G field it comes from
        editor = ValueLineEdit(text, CMS_SOURCE_DECIMALS)
        validator = QDoubleValidator(dialog)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale(QLocale.Language.C))
        editor.setValidator(validator)
        editor.setMaximumWidth(120)
        row.addWidget(editor)
        row.addStretch(1)
        layout.addLayout(row)
        editors[key] = editor

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(420)

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    out = {}
    for key, label, _text, check, why in fields:
        try:
            value = float(editors[key].text().strip().replace(',', '.'))
        except ValueError:
            QMessageBox.warning(main_window, title, f"{label}: please enter a number.")
            return None
        if not check(value):
            QMessageBox.warning(main_window, title, f"{label}: {why}.")
            return None
        out[key] = value
    return out
