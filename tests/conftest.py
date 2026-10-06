"""Shared pytest fixtures for the SYNCmoss test-suite.

The GUI tests run head-less: ``QT_QPA_PLATFORM`` is forced to ``offscreen`` *before*
PySide6 is imported anywhere, so the suite works on CI machines without a display.
"""
import hashlib
import os
import shutil

# Must be set before the very first PySide6 import (including the imports that
# happen transitively when a syncmoss GUI module is imported by a test).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

# The calibration spectrum and the instrumental function the tests compute with:
# frozen copies of syncmoss/parameters/ (as committed), kept with the tests. The
# shipped files are the user's working state -- every calibration rewrites
# Calibration.dat, every instrumental-function search the INS files, and GCMS is
# typed in -- so a test must never depend on what is in them.
FROZEN_PARAMETERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "parameters")
FROZEN_FILES = ("Calibration.dat", "INSexp.txt", "INSint.txt", "INSth.txt", "GCMS.txt")


def redirect_calibration_to_tmp(window, tmp_dir):
    """Point a PhysicsApp at a throw-away copy of the tests' frozen Calibration.dat.

    The GUI tests use it as their spectrum (FROZEN_PARAMETERS: not the shipped
    one, which changes with every calibration). Showing/fitting a spectrum
    re-calibrates and rewrites the spectrum file in place, hence the copy.
    """
    tmp_copy = os.path.join(str(tmp_dir), "Calibration.dat")
    shutil.copy2(os.path.join(FROZEN_PARAMETERS, "Calibration.dat"), tmp_copy)
    window.calibration_path = tmp_copy
    return window.calibration_path


def redirect_params_dir_to_tmp(window, tmp_dir):
    """Point a PhysicsApp at a throw-away copy of ``parameters/`` holding the
    tests' frozen calibration and instrumental function (FROZEN_FILES; the
    GCMS field is set from the frozen GCMS.txt, as the window does at start).

    Every instrumental-function search WRITES its answer: INSexp.txt, INSint.txt
    and (for the theoretical one) INSacc.txt, all under ``app.params_dir``. So a
    test that runs a search silently replaces the shipped instrumental function
    of the package with one fitted to whatever spectrum the test happened to use
    -- no error, no failure, and nothing to restore from if the tree is not under
    version control. This is the same hazard ``redirect_calibration_to_tmp``
    exists for, one directory up. The other files (Be.txt, KB.txt, ...) are the
    shipped ones: tests read them as they are. Calling it again is harmless.
    """
    tmp_copy = os.path.join(str(tmp_dir), "parameters")
    if os.path.abspath(window.params_dir) != os.path.abspath(tmp_copy):
        shutil.copytree(window.params_dir, tmp_copy, dirs_exist_ok=True)
    for name in FROZEN_FILES:
        shutil.copy2(os.path.join(FROZEN_PARAMETERS, name), os.path.join(tmp_copy, name))
    window.params_dir = tmp_copy
    window.GCMS_input.setText(str(np.genfromtxt(os.path.join(tmp_copy, "GCMS.txt"), delimiter="\t")))
    return window.params_dir


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


def _parameters_dir():
    """The SHIPPED parameters folder -- the one no test may write into."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "syncmoss", "parameters")


def _parameters_fingerprint():
    d = _parameters_dir()
    if not os.path.isdir(d):
        return {}
    out = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


@pytest.fixture(scope="session", autouse=True)
def parameters_folder_is_read_only():
    """Fail the session if any test wrote into syncmoss/parameters.

    Several operations WRITE their answer there and there is nothing to restore
    from -- the tree is not always under version control from the test's point
    of view, and the damage is silent:

        * both instrumental-function searches write INSexp.txt / INSint.txt /
          INSth.txt;
        * calibration writes Calibration.dat and calibr.png, and now also
          re-centres the stored instrumental function;
        * "Reset to default values" rewrites whichever description is selected.

    Individual tests are expected to call ``redirect_params_dir_to_tmp`` (and
    ``redirect_calibration_to_tmp``) first. This is the backstop that catches
    the one that forgets, rather than discovering it when the shipped
    instrumental function has quietly become a fit of somebody's test spectrum.
    """
    before = _parameters_fingerprint()
    yield
    after = _parameters_fingerprint()
    changed = sorted(k for k in after if k in before and before[k] != after[k])
    removed = sorted(k for k in before if k not in after)
    created = sorted(k for k in after if k not in before)
    if changed or removed or created:
        raise AssertionError(
            "tests modified the shipped parameters folder -- "
            f"modified={changed} removed={removed} created={created}. "
            "Call redirect_params_dir_to_tmp(window, tmp_path) before anything "
            "that searches, calibrates or resets.")


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
    """A freshly built :class:`PhysicsApp` main window (destroyed on teardown).

    ``calibration_path`` and ``params_dir`` are redirected to temp copies holding
    the tests' frozen calibration and instrumental function, so no test depends
    on the user's current ones -- or writes into them.

    Teardown must DESTROY the window, not just close() it. close() merely hides a
    widget: WA_DeleteOnClose is not set, the window is top-level (no Qt parent to
    delete it), and ParametersTable/ResultsTable/CustomNavigationToolbar all hold
    back-references to it. So a close()-only teardown left the whole window tree
    -- 103 top-level widgets and ~181 MB -- alive for the rest of the session, per
    test. Over the ~110 tests that use this fixture the process reached ~12 GB,
    PhysicsApp construction slowed from 2.6 s to >4.5 s as the widgets piled up,
    and the run eventually stopped making progress altogether.
    deleteLater() alone is NOT enough either: QApplication.processEvents() does
    not deliver DeferredDelete events, so the window is never actually destroyed.

    We destroy with shiboken6.delete() rather than deleteLater() +
    sendPostedEvents(None, DeferredDelete). Both free the window, but
    sendPostedEvents(None, ...) scans the whole posted-event queue for the thread,
    and that scan gets more expensive every time: measured over 16 build/destroy
    cycles the teardown grew 0.75s -> 5.47s (~0.3 s per preceding window), which is
    what made the late GUI tests take 20-27 s each. shiboken6.delete() destroys the
    object immediately with no queue scan and stays flat (0.69s -> 0.90s).
    """
    from syncmoss.syncmoss_main import PhysicsApp

    window = PhysicsApp(pool=None)
    redirect_calibration_to_tmp(window, tmp_path)
    redirect_params_dir_to_tmp(window, tmp_path)
    try:
        yield window
    finally:
        # Run what the window still has queued while it exists. matplotlib's
        # canvas posts its idle redraw with a bare QTimer.singleShot -- showing
        # or resizing a canvas does it, and so does drawing a tall results
        # table -- so left pending it fires on the deleted canvas in whichever
        # LATER test processes events, failing that test with "Internal C++
        # object (FigureCanvasQTAgg) already deleted".
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        window.close()
        window.setParent(None)
        try:
            import shiboken6
            shiboken6.delete(window)
        except Exception:  # pragma: no cover - shiboken6 ships with PySide6
            from PySide6.QtCore import QEvent
            from PySide6.QtWidgets import QApplication
            window.deleteLater()
            QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
