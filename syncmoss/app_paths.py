"""Where the program's own files are: the folder it runs from, and on a frozen
macOS app the per-user folder that takes its writable ones.

Kept light (no numpy, no numba): main.py reads it before anything imports numba
(jit_cache.configure).
"""
import os
import sys

from PySide6.QtCore import QCoreApplication, QStandardPaths


def app_dir():
    """The folder of the program's files (parameters/, Library/, icons/, ...).
    A frozen app keeps its code in _internal/ and its files next to the
    executable; from source or a pip install they are in the package folder."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def is_frozen_macos():
    """True inside the frozen macOS ``.app``: its bundled folders are not
    reliably writable (code-signed, or launched from a read-only mount)."""
    return getattr(sys, 'frozen', False) and sys.platform == 'darwin'


def macos_app_data_dir():
    """The per-user writable location of the frozen macOS app
    (``QStandardPaths.AppDataLocation`` -> ``~/Library/Application Support/SYNCmoss``)."""
    QCoreApplication.setApplicationName('SYNCmoss')  # deterministic AppData path
    target = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not target:
        target = os.path.expanduser('~/Library/Application Support/SYNCmoss')
    return target
