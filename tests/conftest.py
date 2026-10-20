"""Shared pytest fixtures for the SYNCmoss test-suite.

The GUI tests run head-less: ``QT_QPA_PLATFORM`` is forced to ``offscreen`` *before*
PySide6 is imported anywhere, so the suite works on CI machines without a display.
"""
import hashlib
import os
import shutil
import threading

# Must be set before the very first PySide6 import (including the imports that
# happen transitively when a syncmoss GUI module is imported by a test).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Temporary files of the tests go under syncmoss/.pytest_tmp, in numbered run
# folders (pytest-of-<user>/pytest-<N>; pytest keeps the last three). A fixed
# --basetemp is wiped whenever a run starts, so it broke any run going on at
# the same time, e.g. from another chat; numbered folders are locked while used.
_TEMP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".pytest_tmp")
os.makedirs(_TEMP_ROOT, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", _TEMP_ROOT)

import sys

import numpy as np
import pytest

from worker_plan import order_for_workers, plan_workers

# The calibration spectrum and the instrumental function the tests compute with:
# frozen copies of syncmoss/parameters/ (as committed), kept with the tests. The
# shipped files are the user's working state -- every calibration rewrites
# Calibration.dat, every instrumental-function search the INS files, and GCMS is
# typed in -- so a test must never depend on what is in them.
FROZEN_PARAMETERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "parameters")
FROZEN_FILES = ("Calibration.dat", "INSexp.txt", "INSint.txt", "INSth.txt", "GCMS.txt")


class SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


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
    tests' frozen calibration and instrumental function (FROZEN_FILES; the G and
    Δθ fields and the source and aperture diameters are set from the frozen
    GCMS.txt, as the window does at start).

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
    from syncmoss.instrumental_io import read_gcms_file
    g, dtheta, source_diameter, aperture_diameter = read_gcms_file(tmp_copy)
    window.GCMS_input.setText(str(g))
    window.dtheta_input.setText(str(dtheta) if dtheta else "0")
    window.source_diameter_input.setText(str(source_diameter))
    window.aperture_diameter_input.setText(str(aperture_diameter))
    return window.params_dir


# The session ends with a NORMAL interpreter exit, like the application itself
# (main.py: sys.exit(app.exec())). Do not shortcut it with os._exit(): PySide6
# registers an atexit handler that deletes every remaining QObject and then the
# QApplication while Qt is still fully alive, and os._exit skips it.
# On Linux/macOS _exit() is a bare system call, so nothing runs and nothing can
# go wrong. On Windows _exit() is ExitProcess, which still runs every DLL's C++
# static destructors in reverse load order: Qt6Widgets/Qt6Gui tear down their
# globals first; shiboken6 (PySide6 imports it before any Qt DLL) comes last.
# Python was never finalised, so shiboken's BindingManager destructor sees
# Py_IsInitialized() and destroys every wrapper still alive -- the session's
# leftover widgets -- after Qt's globals are gone: access violations in
# QAccessible::queryAccessibleInterface, QThreadStorageData::get,
# pyside6.abi3.dll, ... depending on which object goes first. Until 2026-10 a
# force-exit hook here did exactly that; it was the "Segmentation fault" after
# a fully passing Windows run on the release lane (it had been hidden locally
# by a faulthandler.disable() and by xdist, whose workers exit normally).


# --- parallel runs (pytest-xdist) ----------------------------------------------
# pyproject.toml asks for "-n auto --dist loadgroup --no-loadscope-reorder";
# worker_plan.py says how the number of workers and the order are chosen.

_PLAN = pytest.StashKey[str]()
_WATCH = pytest.StashKey[object]()


class _MemoryWatch:
    """Peak resident memory of this process and its children (the xdist
    workers), sampled every half second and reported at the end of the run."""

    def __init__(self):
        self.thread = None
        try:
            import psutil
        except ImportError:
            return
        self.error = psutil.Error
        self.me = psutil.Process()
        self.peak = 0
        self.children = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, name="memory-watch", daemon=True)
        self.thread.start()

    def run(self):
        while not self.done.wait(0.5):
            try:
                procs = [self.me] + self.me.children(recursive=True)
            except self.error:
                continue
            total = 0
            for p in procs:
                try:
                    total += p.memory_info().rss
                except self.error:
                    pass
            if total > self.peak:
                self.peak, self.children = total, len(procs) - 1

    def stop(self):
        if self.thread is None:
            return ""
        self.done.set()
        self.thread.join()
        return (f"peak memory of the test processes: {self.peak / 2**30:.1f} GB "
                f"(this one and {self.children} child processes)")


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_auto_num_workers(config):
    """The number of workers for -n auto (worker_plan.plan_workers);
    PYTEST_XDIST_AUTO_NUM_WORKERS still overrides it."""
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None
    n, why = plan_workers(config)
    config.stash[_PLAN] = why
    return n


def pytest_configure(config):
    if hasattr(config, "workerinput"):
        return
    if (getattr(config.option, "numprocesses", 0) or 0) > 0:
        # A worker runs one test at a time; an OpenBLAS thread per processor
        # in every worker only oversubscribes the machine (and reserves ~0.7 GB
        # of buffers each). The workers inherit this.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    config.stash[_WATCH] = _MemoryWatch()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    if hasattr(config, "workerinput"):
        order_for_workers(items)


def pytest_sessionstart(session):
    why = session.config.stash.get(_PLAN, None)
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if why and reporter is not None:
        reporter.write_line(why)


def pytest_terminal_summary(terminalreporter, config):
    watch = config.stash.get(_WATCH, None)
    line = watch.stop() if watch is not None else ""
    if line:
        terminalreporter.write_line(line)


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
        * "Reset instrumental function to default values" rewrites GCMS.txt in
          CMS, else whichever SMS description is selected.

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


@pytest.fixture(scope="session", autouse=True)
def channel_average_off():
    """The suite computes with the channel average OFF (Supp -> Average model
    over each channel, ON for the user): every window, and every object without
    the setting, starts with it off, so the tests keep the model they were
    written for. tests/test_channel_average.py turns it on where it is about
    it. Yields the shipped defaults (instrumental_io's, syncmoss_main's)."""
    import syncmoss.instrumental_io as iio
    import syncmoss.syncmoss_main as main
    shipped = (iio.DEFAULT_CHANNEL_AVERAGE, main.DEFAULT_CHANNEL_AVERAGE)
    iio.DEFAULT_CHANNEL_AVERAGE = main.DEFAULT_CHANNEL_AVERAGE = False
    yield shipped
    iio.DEFAULT_CHANNEL_AVERAGE, main.DEFAULT_CHANNEL_AVERAGE = shipped


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
    window = _new_window(tmp_path)
    try:
        yield window
    finally:
        _destroy_window(window)


@pytest.fixture(scope="module")
def shared_physics_app(qapp, tmp_path_factory):
    """ONE window for all the tests of a module that only READ it -- a parser
    fed through the path box, the menus, the shipped documents. A window takes
    ~2 s to build and destroy, so 20 parametrized cases on their own windows
    cost 40 s. Mark those tests ``pytest.mark.xdist_group("<name>")`` as well,
    or a parallel run hands them to many workers and each builds its own.
    A test that presses a button, logs, or changes the table needs its own
    ``physics_app``."""
    window = _new_window(tmp_path_factory.mktemp("shared_window"))
    try:
        yield window
    finally:
        _destroy_window(window)


def _new_window(tmp_dir):
    from syncmoss.syncmoss_main import PhysicsApp

    window = PhysicsApp(pool=None)
    redirect_calibration_to_tmp(window, tmp_dir)
    redirect_params_dir_to_tmp(window, tmp_dir)
    return window


def _destroy_window(window):
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
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
