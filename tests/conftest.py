"""Shared pytest fixtures for the SYNCmoss test-suite.

The GUI tests run head-less: ``QT_QPA_PLATFORM`` is forced to ``offscreen`` *before*
PySide6 is imported anywhere, so the suite works on CI machines without a display.
"""
import os
import shutil

# Must be set before the very first PySide6 import (including the imports that
# happen transitively when a syncmoss GUI module is imported by a test).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest


def redirect_calibration_to_tmp(window, tmp_dir):
    """Point a PhysicsApp at a throw-away copy of Calibration.dat.

    Showing/fitting a spectrum re-calibrates and rewrites the spectrum file in
    place. The GUI tests use the bundled Calibration.dat as the spectrum, so
    without this the tracked data file in the package would be mutated. Copying it
    into a temp dir and repointing ``calibration_path`` keeps the repo clean.
    """
    original = window.calibration_path
    tmp_copy = os.path.join(str(tmp_dir), "Calibration.dat")
    if os.path.exists(original):
        shutil.copy2(original, tmp_copy)
        window.calibration_path = tmp_copy
    return window.calibration_path


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication shared by every GUI test in the session.

    Qt only allows one QApplication per process, so this is session-scoped. The
    test is skipped (rather than failed) if Qt cannot create an offscreen app,
    e.g. when PySide6 is unavailable in the environment.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"PySide6 not available: {exc}")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


@pytest.fixture
def physics_app(qapp, tmp_path):
    """A freshly built :class:`PhysicsApp` main window (closed on teardown).

    ``calibration_path`` is redirected to a temp copy so tests that show/fit the
    bundled Calibration.dat never mutate the tracked data file.
    """
    from syncmoss.syncmoss_main import PhysicsApp

    window = PhysicsApp(pool=None)
    redirect_calibration_to_tmp(window, tmp_path)
    try:
        yield window
    finally:
        window.close()
