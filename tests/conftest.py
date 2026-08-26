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


# Guarantee the test process actually terminates. Head-less ("offscreen") Qt plus
# the widgets/canvases and worker threads the GUI tests spin up can leave a
# native/non-daemon resource alive that blocks interpreter shutdown, so `pytest`
# finishes running and REPORTING every test yet the process never exits (the
# "infinite waiting" noted for this suite — it makes a plain `pytest tests/` look
# hung and forced the foreground / read-the-progress-bar workaround). We record
# the status in sessionfinish and force-exit in `pytest_unconfigure`, which runs
# AFTER the terminal summary + any failure tracebacks are printed but BEFORE the
# interpreter-shutdown hang — so nothing is ever hidden and the real status is
# preserved. Skipped under xdist workers and active pytest-cov so their own
# finalization (IPC / .coverage writing) is never cut short (CI unaffected).
_FORCE_EXIT_STATUS = {}


def pytest_sessionfinish(session, exitstatus):
    _FORCE_EXIT_STATUS["code"] = int(exitstatus)


def pytest_unconfigure(config):
    code = _FORCE_EXIT_STATUS.get("code")
    if code is None:
        return
    if hasattr(config, "workerinput"):
        return  # xdist worker: leave IPC/teardown to xdist
    if config.pluginmanager.hasplugin("pytest_cov") and getattr(config.option, "cov_source", None):
        return  # coverage active: let atexit combine/write .coverage

    import os
    import faulthandler
    import multiprocessing as mp

    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.quit()
    except Exception:
        pass

    # Terminate any lingering multiprocessing pool/workers first, so the force
    # exit below cannot orphan them (orphaned pool processes are a known hazard
    # in this project). Then disable faulthandler so os._exit — which we use to
    # step over the headless-Qt interpreter-shutdown hang — does not print a
    # (harmless) stack dump of the still-live threads.
    for child in mp.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    faulthandler.disable()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


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
