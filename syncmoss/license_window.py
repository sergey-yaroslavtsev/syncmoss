"""Supp -> "License": the license of SYNCmoss and of everything shipped with it.

One window for every licensing text a user may be entitled to see: the MIT
license of SYNCmoss, the third-party notices (NOTICE.txt) and -- in the Windows
and macOS bundles -- the full license text of every third-party component they
contain, i.e. the ``licenses/`` folder that bundle/third_party_licenses.py fills
at build time.

Run from a Python installation (a source checkout, or pip), the third-party
packages were installed separately; the window then shows the license files the
libraries named in NOTICE.txt were installed with.
"""
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QLabel, QListWidget, QMainWindow, QPlainTextEdit, QSplitter, QVBoxLayout, QWidget,
)

MISSING = "(not found in this installation)"
# The libraries NOTICE.txt names: title -> their distributions
NOTICE_LIBRARIES = {
    'PySide6': ('PySide6_Essentials', 'shiboken6'),
    'NumPy': ('numpy',),
    'SciPy': ('scipy',),
    'Matplotlib': ('matplotlib',),
    'Numba': ('numba',),
    'llvmlite': ('llvmlite',),
}
# PySide6's wheels carry only a Qt commercial-license notice; the terms it is
# used under are these, committed in bundle/licenses/ of a source checkout
GNU_TEXTS = ('LGPL-3.0.txt', 'GPL-3.0.txt')
_LICENSE_NAMES = ('LICEN', 'COPYING', 'NOTICE', 'AUTHORS')


def _read(path):
    with open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()


def _own_file(dir_path, name):
    """Text of SYNCmoss's own LICENSE / NOTICE.txt, or None.

    It is next to the executable in the bundles (``dir_path``), at the root of a
    source checkout, and in the package metadata of a pip installation.
    """
    folders = [dir_path]
    root = os.path.dirname(dir_path)
    if os.path.isfile(os.path.join(root, 'pyproject.toml')):  # a source checkout
        folders.append(root)
    for folder in folders:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return _read(path)
    try:
        dist = distribution('syncmoss')
    except PackageNotFoundError:
        return None
    for file in dist.files or []:
        if file.name == name and file.parts[0].endswith('.dist-info'):
            return file.read_text(encoding='utf-8')
    return None


def bundled_licenses_dir():
    """The bundle's ``licenses/`` folder, or None when not running from one."""
    if not getattr(sys, 'frozen', False):
        return None
    path = os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)), 'licenses')
    return path if os.path.isdir(path) else None


def _files_text(paths, base):
    """The files, each headed by its path relative to *base*."""
    return '\n\n'.join(
        f"==== {os.path.relpath(path, base).replace(os.sep, '/')} ====\n\n{_read(path).rstrip()}\n"
        for path in paths)


def _folder_files(folder):
    found = []
    for current, dirs, files in os.walk(folder):
        dirs.sort()
        found += [os.path.join(current, name) for name in sorted(files)]
    return found


def _installed_license_text(dist_names):
    """The license files these installed distributions came with ('' if none)."""
    parts = []
    for dist_name in dist_names:
        try:
            dist = distribution(dist_name)
        except PackageNotFoundError:
            continue
        for file in dist.files or []:
            if not file.parts[0].endswith('.dist-info') or len(file.parts) < 2:
                continue
            if file.parts[1] != 'licenses' and not file.name.upper().startswith(_LICENSE_NAMES):
                continue
            inside = '/'.join(file.parts[1:])
            parts.append(f"==== {dist_name}: {inside} ====\n\n{_read(str(file.locate())).rstrip()}\n")
    return '\n\n'.join(parts)


def _gnu_texts(dir_path):
    """The LGPL/GPL texts from a source checkout, else where to read them."""
    folder = os.path.join(os.path.dirname(dir_path), 'bundle', 'licenses')
    paths = [os.path.join(folder, name) for name in GNU_TEXTS
             if os.path.isfile(os.path.join(folder, name))]
    if paths:
        return _files_text(paths, folder)
    return "PySide6 is used under the LGPL v3.0: https://www.gnu.org/licenses/lgpl-3.0.html"


def _python_license():
    # python.org: LICENSE.txt in the install folder (Windows) or the standard
    # library (macOS, Linux); conda: LICENSE_PYTHON.txt in the environment
    for candidate in (os.path.join(sys.base_prefix, 'LICENSE.txt'),
                      os.path.join(os.path.dirname(os.__file__), 'LICENSE.txt'),
                      os.path.join(sys.base_prefix, 'LICENSE_PYTHON.txt')):
        if os.path.isfile(candidate):
            return _read(candidate)
    return f"{MISSING}; see https://docs.python.org/3/license.html"


def license_sections(dir_path):
    """[(title, text)]: SYNCmoss's license, the notices, then every component."""
    sections = [("SYNCmoss - MIT License", _own_file(dir_path, 'LICENSE') or MISSING),
                ("Third-party notices", _own_file(dir_path, 'NOTICE.txt') or MISSING)]
    folder = bundled_licenses_dir()
    if folder is None:  # from Python: the libraries NOTICE.txt names
        for title, dist_names in NOTICE_LIBRARIES.items():
            text = _installed_license_text(dist_names)
            if title == 'PySide6':  # the terms it is used under first
                text = '\n\n'.join(part for part in (_gnu_texts(dir_path), text) if part)
            if text:
                sections.append((title, text))
        sections.append(("Python", _python_license()))
        return sections

    entries = sorted(os.listdir(folder), key=str.lower)
    for name in entries:  # one folder per component
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            sections.append((name, _files_text(_folder_files(path), path)))
    for name in entries:  # loose texts: LGPL-3.0.txt, GPL-3.0.txt
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            sections.append((os.path.splitext(name)[0], _files_text([path], folder)))
    return sections


class LicenseWindow(QMainWindow):
    """The :func:`license_sections`: a list on the left, the chosen text on the right."""

    def __init__(self, dir_path, parent=None):
        super().__init__(parent)
        self.dir_path = dir_path
        self.sections = []
        # A standalone top-level window (taskbar entry), like the document viewers
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle("SYNCmoss - License")
        self.resize(1100, 780)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        intro = QLabel(
            "SYNCmoss is free software under the MIT License. It uses the "
            "third-party components listed below, each under its own license; "
            "their full texts are here.", self)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.index = QListWidget(splitter)
        self.viewer = QPlainTextEdit(splitter)
        self.viewer.setReadOnly(True)
        self.viewer.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 840])
        layout.addWidget(splitter, 1)

        self.index.currentRowChanged.connect(self._show_section)
        self.reload()

    def reload(self):
        """Re-read every text and show the first one."""
        self.sections = license_sections(self.dir_path)
        self.index.clear()
        self.index.addItems([title for title, _ in self.sections])
        self.index.setCurrentRow(0)

    def _show_section(self, row):
        if 0 <= row < len(self.sections):
            self.viewer.setPlainText(self.sections[row][1])
