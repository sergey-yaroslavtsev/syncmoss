"""Calibration.dat is READ from where it is WRITTEN.

The calibration procedure writes into ``params_dir`` -- ``CalibrationThread`` is
constructed with it -- but ``load_spectrum``'s default was the bare name
"Calibration.dat", which it resolved against ``dir_path``, the working folder.
Any caller that omitted the argument therefore calibrated against a stale file
in the working folder, or against nothing at all, and said nothing about it:
the spectrum still loads, just on the wrong velocity axis. Fitting did exactly
that, at both of its call sites.

Those two are fixed, but the default is what made it possible, so the default
is what these tests pin.
"""
import inspect
import os

import pytest

from syncmoss.spectrum_io import resolve_calibration_path


def test_default_is_the_windows_own_path(physics_app):
    """The bug: this used to land in dir_path.

    Asserted against ``calibration_path`` rather than against params_dir
    literally, because the test fixture redirects it to a temp copy -- and
    following the window wherever it points is precisely the property that was
    missing.
    """
    got = resolve_calibration_path(physics_app, None, ['/data/spec.dat'])
    assert got == physics_app.calibration_path


def test_default_is_not_the_working_folder(physics_app):
    got = resolve_calibration_path(physics_app, None, ['/data/spec.dat'])
    assert got != os.path.join(physics_app.dir_path, "Calibration.dat")


def test_an_explicit_absolute_path_wins(physics_app, tmp_path):
    """Callers that pass one -- most of them -- are unaffected."""
    explicit = str(tmp_path / "Other.dat")
    assert resolve_calibration_path(physics_app, explicit, ['/d/s.dat']) \
        == explicit


def test_an_explicit_relative_name_is_still_window_relative(physics_app):
    """Unchanged behaviour: a bare name given ON PURPOSE means the working
    folder, which is what the few callers that do it expect."""
    got = resolve_calibration_path(physics_app, "Mine.dat", ['/d/s.dat'])
    assert got == os.path.join(physics_app.dir_path, "Mine.dat")


def test_without_a_window_it_falls_back_to_the_spectrum_folder(tmp_path):
    """Head-less callers -- ``calculate_backgrounds`` passes main_window=None --
    keep the old relative behaviour, since there is no window to ask."""
    spectrum = str(tmp_path / "spec.dat")
    assert resolve_calibration_path(None, None, [spectrum]) \
        == os.path.join(str(tmp_path), "Calibration.dat")


def test_calibration_is_written_where_it_is_read():
    """The two halves must agree. CalibrationThread is handed the directory the
    file is written into, and the window points calibration_path at the same
    one; the tests' fixture then redirects BOTH together, which is why this is
    checked on the source rather than on a live window."""
    from syncmoss import syncmoss_main
    src = ' '.join(inspect.getsource(syncmoss_main.PhysicsApp).split())
    i = src.index("CalibrationThread(")
    assert "self.params_dir" in src[i:i + 120], \
        "calibration is no longer written into params_dir"
    j = src.index("self.calibration_path = ")
    assert "self.params_dir" in src[j:j + 90], \
        "calibration_path no longer points into params_dir"


def test_fitting_passes_the_path_explicitly():
    """Belt and braces: the two sites that caused this keep passing it."""
    from syncmoss import fitting_io
    src = ' '.join(inspect.getsource(fitting_io).split())
    for call in ("load_spectrum( app, spec_file",
                 "load_spectrum(app, spectrum_file"):
        i = src.find(call)
        assert i >= 0, f"call site moved: {call}"
        assert "calibration_path" in src[i:i + 160], call


@pytest.mark.parametrize("module_name", ["fitting_io", "instrumental_io",
                                         "spectrum_io", "syncmoss_main",
                                         "parameters_table"])
def test_no_module_hardcodes_the_bare_name(module_name):
    """Nothing should join 'Calibration.dat' onto dir_path again."""
    import importlib
    src = inspect.getsource(importlib.import_module(f"syncmoss.{module_name}"))
    assert "dir_path, 'Calibration.dat'" not in src
    assert 'dir_path, "Calibration.dat"' not in src
