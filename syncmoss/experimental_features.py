"""New features that are not well tested yet.

A feature named in EXPERIMENTAL_FEATURES has its model-menu entry in
EXPERIMENTAL_COLOR (parameters_table), and choosing it shows
warn_if_experimental's message first. Once a feature has proven itself, it is
taken off the list and nothing else changes: its entry gets its usual colour
back and the message is no longer shown.
"""
from PySide6.QtWidgets import QMessageBox

EXPERIMENTAL_FEATURES = ('Distr_T', 'Spectrum')

EXPERIMENTAL_COLOR = '#cc0000'


def warn_if_experimental(parent, feature):
    """Tell the user that *feature* is new and not well tested, if it is on the list."""
    if feature not in EXPERIMENTAL_FEATURES:
        return
    QMessageBox.warning(
        parent, f"{feature}: new feature",
        f"'{feature}' is a new feature and is not well tested yet. "
        "Please be careful while using it.\n\n"
        "If you find a bug, please report it (Supp → Contact the author).")
