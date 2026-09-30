"""Supp -> License, and the license texts the bundles carry.

The window lists SYNCmoss's MIT license, the third-party notices and -- in a
Windows/macOS bundle -- every text in the bundle's ``licenses/`` folder, which
bundle/third_party_licenses.py fills at build time from what PyInstaller
actually packed. Pinned:

  * from a source checkout: the license, the notices, and the license files of
    the libraries NOTICE.txt names, from the installed packages;
  * in a bundle: one entry per component folder, plus the loose GNU texts;
  * the Supp entry opens the window;
  * the build helper collects the license files of every bundled package, and
    the two GNU texts no wheel provides are committed next to it.
"""
import hashlib
import importlib.util
import os
import sys

import pytest

from syncmoss import license_window as lw

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_from_a_source_checkout_it_shows_the_notice_libraries():
    # dir_path as PhysicsApp sets it when not frozen: the package folder
    sections = lw.license_sections(os.path.join(_REPO, "syncmoss"))
    assert [title for title, _ in sections] == [
        "SYNCmoss - MIT License", "Third-party notices",
        "PySide6", "NumPy", "SciPy", "Matplotlib", "Numba", "llvmlite", "Python"]
    text = dict(sections)
    assert "MIT License" in text["SYNCmoss - MIT License"] and "ESRF" in text["SYNCmoss - MIT License"]
    assert "Third-Party Software Notices" in text["Third-party notices"]
    # PySide6's wheels carry only a commercial notice: the LGPL comes from bundle/licenses/
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text["PySide6"]
    assert "NumPy Developers" in text["NumPy"]
    assert "Anaconda" in text["Numba"]


def test_in_a_bundle_it_shows_every_license_text(monkeypatch, tmp_path):
    (tmp_path / "LICENSE").write_text("MIT License\nCopyright (c) ESRF\n")
    (tmp_path / "NOTICE.txt").write_text("the notices\n")
    licenses = tmp_path / "_internal" / "licenses"
    (licenses / "numba").mkdir(parents=True)
    (licenses / "numba" / "LICENSE").write_text("numba is BSD\n")
    (licenses / "numpy" / "numpy" / "random").mkdir(parents=True)
    (licenses / "numpy" / "numpy" / "random" / "LICENSE.md").write_text("random is BSD\n")
    (licenses / "LGPL-3.0.txt").write_text("GNU LESSER GENERAL PUBLIC LICENSE\n")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)

    sections = lw.license_sections(str(tmp_path))  # dir_path: next to the executable
    titles = [title for title, _ in sections]
    assert titles == ["SYNCmoss - MIT License", "Third-party notices",
                      "numba", "numpy", "LGPL-3.0"]
    text = dict(sections)
    assert "Copyright (c) ESRF" in text["SYNCmoss - MIT License"]
    assert "the notices" in text["Third-party notices"]
    assert "numba is BSD" in text["numba"]
    assert "==== numpy/random/LICENSE.md ====" in text["numpy"]
    assert "GNU LESSER" in text["LGPL-3.0"]


@pytest.mark.gui
def test_supp_license_opens_the_window(physics_app):
    """Destroyed again before the test ends: a stray top-level widget left alive
    head-less is what makes this suite hang at interpreter shutdown."""
    import syncmoss.supp_menu as supp_menu
    from PySide6.QtWidgets import QApplication

    w = physics_app
    assert "License" in [action.text() for action in w.supp_menu.actions()]
    assert w.license_window is None
    try:
        supp_menu.open_license_pressed(w)
        window = w.license_window
        assert window.index.count() == len(window.sections) >= 3
        assert "MIT License" in window.viewer.toPlainText()
        window.index.setCurrentRow(1)
        assert "Third-Party Software Notices" in window.viewer.toPlainText()
        supp_menu.open_license_pressed(w)  # a second call reuses the window
        assert w.license_window is window
    finally:
        window = w.license_window
        w.license_window = None
        if window is not None:
            window.close()
            window.setParent(None)
            window.deleteLater()
        QApplication.processEvents()


# --- the build side -----------------------------------------------------------

def _bundle_module(name):
    path = os.path.join(_REPO, "bundle", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name, sha256", [
    ("LGPL-3.0.txt", "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118"),
    ("GPL-3.0.txt", "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"),
])
def test_the_gnu_texts_are_the_official_ones(name, sha256):
    """PySide6's wheels ship only a Qt commercial-license notice, so the two
    texts the LGPL requires are committed -- verbatim from gnu.org."""
    with open(os.path.join(_REPO, "bundle", "licenses", name), "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == sha256


@pytest.mark.skipif(sys.version_info < (3, 10), reason="packages_distributions() is 3.10+")
def test_the_build_collects_the_texts_of_what_is_bundled():
    tpl = _bundle_module("third_party_licenses")

    class Analysis:  # what PyInstaller's Analysis offers the helper
        pure = [("numpy", "", "PYMODULE"), ("numba.core", "", "PYMODULE"), ("os", "", "PYMODULE")]
        binaries = [(os.path.join("PySide6", "Qt6Core.dll"), "", "BINARY")]
        datas = []

    entries = tpl.license_datas(Analysis)
    destinations = {dest.replace("\\", "/") for dest, _, _ in entries}
    assert any(d.startswith("licenses/numpy/") for d in destinations)
    assert any(d.startswith("licenses/numba/") for d in destinations)
    assert {"licenses/LGPL-3.0.txt", "licenses/GPL-3.0.txt"} <= destinations
    assert all(os.path.isfile(src) for _, src, _ in entries)
    assert all(kind == "DATA" for _, _, kind in entries)
