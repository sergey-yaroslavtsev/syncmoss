"""Multispectra settings -> Fit one spectrum: which one.

One spectrum of the path box -- or a file next to them -- is fitted on its own
with the model of the table. The dialog asks for it by its number (its place in
the path box, counted from 1) or by its name (the file name without .dat), and
each field follows the other: a number puts in its spectrum's name, the name of
a spectrum of the path box puts in its number ("—" for any other name).
"""
import os

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QVBoxLayout,
)

from syncmoss.model_io import strip_known_extension


def spectrum_name(path):
    """A spectrum's name: its file name without the extension."""
    return strip_known_extension(os.path.basename(path))


def find_spectrum(entries, name):
    """The spectrum *name* stands for, as ``(number, values, path)``.

    A spectrum of the path box *entries* (``(path, values)``) by its name --
    typed with or without its extension -- comes with its number and values.
    Otherwise a file of that name next to the spectra of the path box (in their
    folders, with their extensions or the one typed) counts, with no number and
    no values. None when there is neither.
    """
    typed = name.strip()
    wanted = strip_known_extension(typed)
    if not wanted:
        return None
    for number, (path, values) in enumerate(entries, start=1):
        if spectrum_name(path) == wanted:
            return number, tuple(values), path
    extensions = ([typed[len(wanted):]] if typed != wanted
                  else list(dict.fromkeys(os.path.splitext(path)[1] for path, _ in entries)))
    for folder in dict.fromkeys(os.path.dirname(os.path.abspath(path)) for path, _ in entries):
        for extension in extensions:
            candidate = os.path.join(folder, wanted + extension)
            if os.path.isfile(candidate):
                return None, (), candidate
    return None


class OneSpectrumDialog(QDialog):
    """Which spectrum to fit: its number or its name, each following the other.

    After OK, ``chosen`` is :func:`find_spectrum`'s answer; OK does not close
    the dialog while the name is neither a spectrum of the path box nor a file
    next to them.
    """

    def __init__(self, parent, entries):
        super().__init__(parent)
        self.setWindowTitle("Fit one spectrum")
        self.entries = list(entries)
        self.names = [spectrum_name(path) for path, _ in self.entries]
        self.chosen = None

        self.number = QSpinBox(self)
        self.number.setRange(0, len(self.entries))
        self.number.setSpecialValueText("—")         # a name not in the path box
        self.number.setValue(1)
        self.name = QLineEdit(self.names[0], self)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.hide()

        form = QFormLayout()
        form.addRow(f"Number in the path box (1–{len(self.entries)}):", self.number)
        form.addRow("Name (the file name without .dat):", self.name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.message)
        layout.addWidget(buttons)

        self.number.valueChanged.connect(self.number_changed)
        self.name.textEdited.connect(self.name_changed)

    def number_changed(self, number):
        """A number puts in its spectrum's name."""
        if number:
            self.name.setText(self.names[number - 1])

    def name_changed(self, text):
        """The name of a spectrum of the path box puts in its number, any
        other name "—"."""
        wanted = strip_known_extension(text.strip())
        self.number.blockSignals(True)
        self.number.setValue(self.names.index(wanted) + 1 if wanted in self.names else 0)
        self.number.blockSignals(False)

    def accept(self):
        found = find_spectrum(self.entries, self.name.text())
        if found is None:
            self.message.setText(f"There is no spectrum '{self.name.text().strip()}' in the "
                                 f"path box, nor next to its spectra.")
            self.message.show()
            return
        self.chosen = found
        super().accept()
