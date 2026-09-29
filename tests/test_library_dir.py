"""The model Library gets the same frozen-macOS redirect as ``parameters/``.

A frozen macOS ``.app`` is code-signed / launched from a read-only mount, so
``_resolve_params_dir`` keeps the parameter files in
``~/Library/Application Support/SYNCmoss``. "Save to library" and "Import
Library" write too, but they wrote into the bundled ``Library/`` next to the
executable -- inside the app. The Library now takes the same route,
``<AppData>/Library``, seeded from the bundled models by the same rule, and every
call site reads ``main_window.library_dir`` instead of rebuilding the path from
``dir_path``.
"""
import importlib
import inspect
import os
import sys

import pytest

from syncmoss import syncmoss_main as sm
from syncmoss import supp_menu
from syncmoss import Library_window
from syncmoss import model_io

MODEL = "#@Chemical composition Fe2O3\n#@Temperature (K) 300\nbaseline\tSextet\n"


def _write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _app_folder(tmp_path):
    """What ships next to the frozen executable: parameters/ and Library/."""
    base = tmp_path / "SYNCmoss.app" / "Contents" / "MacOS"
    for folder, names in (("parameters", ("Be.txt", "Calibration.dat")),
                          ("Library", ("Goethite.mdl", "Hematite.mdl"))):
        os.makedirs(base / folder)
        for name in names:
            _write(base / folder / name, f"bundled {name}\n")
    return str(base)


@pytest.fixture
def frozen_macos(tmp_path, monkeypatch):
    """Run the resolvers as the frozen macOS app does, with AppData in tmp_path."""
    app_data = str(tmp_path / "Application Support" / "SYNCmoss")
    monkeypatch.setattr(sm, "_is_frozen_macos", lambda: True)
    monkeypatch.setattr(sm, "_macos_app_data_dir", lambda: app_data)
    return app_data


# --- where the folders resolve to ------------------------------------------

def test_only_a_frozen_macos_app_is_redirected(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert sm._is_frozen_macos()
    monkeypatch.setattr(sys, "platform", "win32")
    assert not sm._is_frozen_macos()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delattr(sys, "frozen")
    assert not sm._is_frozen_macos()


def test_source_and_windows_runs_use_the_folders_in_place(tmp_path, monkeypatch):
    base = _app_folder(tmp_path)
    app_data = tmp_path / "Application Support"
    monkeypatch.setattr(sm, "_is_frozen_macos", lambda: False)
    monkeypatch.setattr(sm, "_macos_app_data_dir", lambda: str(app_data / "SYNCmoss"))

    assert sm._resolve_params_dir(base) == os.path.join(base, "parameters")
    assert sm._resolve_library_dir(base) == os.path.join(base, "Library")
    assert not app_data.exists()


def test_frozen_macos_moves_both_folders_to_app_data(tmp_path, frozen_macos):
    base = _app_folder(tmp_path)
    params_dir = sm._resolve_params_dir(base)
    library_dir = sm._resolve_library_dir(base)

    assert params_dir == frozen_macos
    assert library_dir == os.path.join(frozen_macos, "Library")
    assert sorted(os.listdir(library_dir)) == ["Goethite.mdl", "Hematite.mdl"]
    assert _read(os.path.join(library_dir, "Hematite.mdl")) == "bundled Hematite.mdl\n"
    # the Library is a folder of its own in AppData, not mixed into the
    # parameter files
    assert sorted(os.listdir(params_dir)) == ["Be.txt", "Calibration.dat", "Library"]


@pytest.mark.parametrize("folder, resolver, bundled_file, own_file, new_file", [
    ("parameters", "_resolve_params_dir", "Calibration.dat", "notes.txt", "INSnew.txt"),
    ("Library", "_resolve_library_dir", "Hematite.mdl", "My sample.mdl", "Wuestite.mdl"),
])
def test_changes_are_kept_until_the_bundle_gains_a_file(
        tmp_path, frozen_macos, folder, resolver, bundled_file, own_file, new_file):
    """The rule both folders share: while every bundled file exists in AppData
    the user's changes are kept; a missing one means the software logic
    changed, so ALL bundled files are copied again. The user's own files
    survive both."""
    base = _app_folder(tmp_path)
    resolve = getattr(sm, resolver)
    target = resolve(base)
    _write(os.path.join(target, bundled_file), "changed by the user\n")
    _write(os.path.join(target, own_file), "the user's own\n")

    assert resolve(base) == target  # next launch of the same version
    assert _read(os.path.join(target, bundled_file)) == "changed by the user\n"

    _write(os.path.join(base, folder, new_file), "new in this version\n")
    assert resolve(base) == target  # first launch of a version with one more file
    assert _read(os.path.join(target, new_file)) == "new in this version\n"
    assert _read(os.path.join(target, bundled_file)) == f"bundled {bundled_file}\n"
    assert _read(os.path.join(target, own_file)) == "the user's own\n"


# --- every call site uses main_window.library_dir ---------------------------

class _Window:
    """The main-window attributes the Library call sites read. ``dir_path``
    holds no Library at all, so a call site that still builds the path from it
    fails instead of passing by accident."""

    def __init__(self, tmp_path):
        self.dir_path = str(tmp_path / "app")
        self.library_dir = str(tmp_path / "user data" / "Library")
        self.workfolder = str(tmp_path / "work")
        self.messages = []
        os.makedirs(self.dir_path)
        os.makedirs(self.library_dir)
        os.makedirs(self.workfolder)

    def set_status(self, message, color=None):
        self.messages.append(message)


class _FolderDialog:
    """Stands in for QFileDialog: answers ``folder`` and records where it opened."""

    def __init__(self, folder):
        self.folder = folder
        self.opened_in = []

    def getExistingDirectory(self, parent, caption, start):
        self.opened_in.append(start)
        return self.folder


class _SilentMessageBox:
    def information(self, *args):
        pass


def test_export_copies_the_library_dir(tmp_path, monkeypatch):
    window = _Window(tmp_path)
    _write(os.path.join(window.library_dir, "Mine.mdl"), MODEL)
    destination = tmp_path / "USB stick"
    destination.mkdir()
    dialog = _FolderDialog(str(destination))
    monkeypatch.setattr(supp_menu, "QFileDialog", dialog)

    supp_menu.export_library_pressed(window)

    assert window.messages[-1].startswith("Library exported to:"), window.messages
    assert os.listdir(destination / "Library") == ["Mine.mdl"]
    # the dialog opens in the Library folder itself, not in the work folder
    assert dialog.opened_in == [window.library_dir]


def test_import_writes_into_the_library_dir(tmp_path, monkeypatch):
    window = _Window(tmp_path)
    shared = tmp_path / "from a colleague"
    shared.mkdir()
    _write(shared / "Theirs.mdl", MODEL)
    dialog = _FolderDialog(str(shared))
    monkeypatch.setattr(supp_menu, "QFileDialog", dialog)
    monkeypatch.setattr(supp_menu, "QMessageBox", _SilentMessageBox())

    supp_menu.import_library_pressed(window)

    assert window.messages == ["Imported 1 .mdl file(s) into Library"]
    assert os.listdir(window.library_dir) == ["Theirs.mdl"]
    assert os.listdir(window.dir_path) == []
    assert dialog.opened_in == [window.library_dir]


def test_the_browser_reads_the_library_dir(tmp_path):
    window = _Window(tmp_path)
    window.library_dir = str(tmp_path / "not there")

    Library_window.open_library_model_dialog(window, None, 0)

    assert window.messages == [f"Library folder not found: {window.library_dir}"]


@pytest.mark.gui
def test_the_window_saves_into_its_library_dir(physics_app, tmp_path):
    # a source run keeps the bundled folder in place, exactly like params_dir
    assert physics_app.library_dir == os.path.join(physics_app.dir_path, "Library")

    # both redirected into tmp_path, so a call site that fell back to dir_path
    # fails here without writing into the shipped Library
    physics_app.dir_path = str(tmp_path / "app")
    physics_app.library_dir = str(tmp_path / "user data" / "Library")

    assert model_io.save_model_to_library(physics_app, "Probe sample")
    assert os.listdir(physics_app.library_dir) == ["Probe sample.mdl"]
    assert not os.path.exists(physics_app.dir_path)


@pytest.mark.gui
def test_a_model_with_nbaseline_is_not_saved(physics_app, tmp_path):
    """The Library takes no models with Nbaseline; those go through Save model
    and the 'Load model' dropdown entry instead."""
    physics_app.dir_path = str(tmp_path / "app")
    physics_app.library_dir = str(tmp_path / "user data" / "Library")
    pt = physics_app.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Nbaseline")
    pt.select_model(3, "Sextet")

    assert model_io.save_model_to_library(physics_app, "With Nbaseline") is False
    assert physics_app.log.toPlainText() == "Model with 'Nbaseline' could not be saved to library"
    assert not os.path.exists(physics_app.library_dir)
    assert not os.path.exists(physics_app.dir_path)


@pytest.mark.parametrize("module_name", ["model_io", "Library_window", "supp_menu",
                                         "syncmoss_main", "parameters_table"])
def test_no_module_builds_the_library_path_from_dir_path(module_name):
    """Nothing should join 'Library' onto dir_path again: that is the bundled
    folder, not necessarily the one the window reads and writes."""
    src = inspect.getsource(importlib.import_module(f"syncmoss.{module_name}"))
    assert "dir_path, 'Library'" not in src
    assert 'dir_path, "Library"' not in src
