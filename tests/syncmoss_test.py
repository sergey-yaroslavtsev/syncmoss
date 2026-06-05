"""GUI smoke / integration test for SYNCmoss.

The SAME flow can be driven two ways:

* ``pytest tests``     -> runs ``test_gui_open_and_fit`` (head-less, ThreadPool)
* ``syncmoss --test``  -> calls ``run_gui_smoke`` (real multiprocessing pool),
                          including inside the frozen .exe / .app binaries.

Both share the ``_TestRunner`` driver below. This module lives in ``tests/`` and is
the only test helper bundled into the frozen binaries (see ``bundle/*.spec``), so
``--test`` works there without pulling pytest into the bundle.

The flow:
  1. open the GUI
  2. show a spectrum (Calibration.dat = an alpha-iron calibration)
  3. refine the instrumental function (pure alpha-iron)
  4. switch to a TWO-spectrum path ([Calibration.dat, Calibration.dat])
  5. build the model: add Sextet, Doublet, Nbaseline; delete the Doublet; Insert a
     row before Nbaseline; turn that row into a Be model; add a Sextet after
     Nbaseline  ->  Sextet, Be(=Doublet), Nbaseline, Sextet
  6. show the model
  7. fit (simultaneous, because of Nbaseline + two spectra)
  8. assert reduced chi-square < 2
  9. assert the per-model colours match between the parameters table and the
     results table (and equal the expected red-cyan-cyan-yellow)

Exit code of ``run_gui_smoke``: 0 on success, 1 on any failure.
"""
import os
import re
import shutil
import sys
import tempfile
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from syncmoss.syncmoss_main import PhysicsApp

# pytest is NOT bundled into the frozen binaries; import it lazily so that
# `syncmoss --test` can import this module without it.
try:
    import pytest
except ImportError:  # pragma: no cover - only in frozen binaries
    pytest = None

if pytest is not None:
    pytestmark = [pytest.mark.gui, pytest.mark.slow]


# How often to re-check whether an async operation has finished (ms)
_POLL_MS = 500
# Max polls before giving up on one async step (~60 s at 500 ms each)
_TIMEOUT_POLLS = 120

# Expected final model component colours (baseline excluded). The user verified
# these are produced by the delete/insert/Be sequence below; the key invariant is
# that the parameters table and the results table agree on them.
_EXPECTED_MODEL_COLORS = ['red', 'cyan', 'cyan', 'yellow']
# get_model_list() reports a Be model as a "Doublet" (it is a Doublet preset).
_EXPECTED_MODEL_LIST = ['baseline', 'Sextet', 'Doublet', 'Nbaseline', 'Sextet']

# Parameter files the instrumental refinement rewrites in <dir_path>/parameters/.
# They are snapshotted before the run and restored afterwards so the test never
# leaves the tracked data files modified.
_PROTECTED_PARAM_FILES = ('INSexp.txt', 'INSint.txt', 'GCMS.txt')


def _parse_bg_color(stylesheet: str):
    """Extract the ``background-color`` value from a Qt stylesheet string."""
    match = re.search(r'background-color:\s*([^;]+);', stylesheet or '')
    return match.group(1).strip() if match else None


def _use_temp_calibration(window) -> None:
    """Point the window at a throw-away copy of Calibration.dat.

    Showing/fitting a spectrum re-calibrates and rewrites the spectrum file in
    place; the smoke test uses the bundled Calibration.dat as the spectrum, so
    without this it would mutate the installed/tracked data file.
    """
    src = window.calibration_path
    if src and os.path.exists(src):
        tmp_dir = tempfile.mkdtemp(prefix="syncmoss_smoke_")
        dst = os.path.join(tmp_dir, "Calibration.dat")
        shutil.copy2(src, dst)
        window.calibration_path = dst


def _snapshot_param_files(window) -> dict:
    """Read the protected parameter files so they can be restored after the run."""
    snapshot = {}
    params_dir = os.path.join(window.dir_path, 'parameters')
    for name in _PROTECTED_PARAM_FILES:
        path = os.path.join(params_dir, name)
        snapshot[path] = None
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                snapshot[path] = fh.read()
    return snapshot


def _restore_param_files(snapshot: dict) -> None:
    """Restore (or remove) the protected parameter files from a snapshot."""
    for path, content in snapshot.items():
        if content is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            with open(path, 'wb') as fh:
                fh.write(content)


class _TestRunner:
    """Drives the GUI through the smoke-test steps sequentially.

    Each synchronous step schedules the next with a short ``QTimer``; each async
    step (instrumental refine / show model / fit) is followed by ``_wait`` which
    polls the ``inprogress`` flag until the background QThread finishes.
    """

    def __init__(self, app: QApplication, window: PhysicsApp) -> None:
        self.app = app
        self.window = window
        self.errors: list = []
        self._poll_count = 0
        self._next_step = None
        self._wait_label = ""

    # -- entry point ---------------------------------------------------

    def start(self) -> None:
        QTimer.singleShot(300, self.step_show_spectrum)

    # Backwards-compatible alias (older callers used step_add_sextet as entry).
    step_add_sextet = start

    # -- generic async wait -------------------------------------------

    def _wait(self, next_step, label) -> None:
        self._poll_count = 0
        self._next_step = next_step
        self._wait_label = label
        QTimer.singleShot(_POLL_MS, self._poll)

    def _poll(self) -> None:
        if not self.window.inprogress:
            if "red" in self.window.log.styleSheet():
                self.errors.append(f"{self._wait_label}: {self.window.log.toPlainText()}")
                self._finish()
                return
            print(f"[TEST] {self._wait_label} OK")
            QTimer.singleShot(300, self._next_step)
        elif self._poll_count >= _TIMEOUT_POLLS:
            self.errors.append(f"{self._wait_label}: timeout")
            self._finish()
        else:
            self._poll_count += 1
            QTimer.singleShot(_POLL_MS, self._poll)

    # -- Step 1: show spectrum (synchronous) --------------------------

    def step_show_spectrum(self) -> None:
        print("[TEST] Step: show spectrum (Calibration.dat) ...")
        w = self.window
        cal = w.calibration_path
        if not os.path.exists(cal):
            self.errors.append(f"show spectrum: Calibration.dat not found at '{cal}'")
            self._finish()
            return
        w.process_path.setPlainText(repr([cal]))
        w.show_pressed()  # synchronous
        if "red" in w.log.styleSheet():
            self.errors.append(f"show spectrum: {w.log.toPlainText()}")
            self._finish()
            return
        print("[TEST] show spectrum OK")
        QTimer.singleShot(300, self.step_refine_instrumental)

    # -- Step 2: refine instrumental function, pure alpha-iron (async) -

    def step_refine_instrumental(self) -> None:
        print("[TEST] Step: refine instrumental function (pure a-Fe) ...")
        self.window.instrumental_pressed(1, 2)  # ref=1 (refine), mode=2 (pure a-Fe)
        self._wait(self.step_set_two_paths, "refine instrumental")

    # -- Step 3: switch to a two-spectrum path (synchronous) ----------

    def step_set_two_paths(self) -> None:
        print("[TEST] Step: set two-spectrum path ...")
        w = self.window
        cal = w.calibration_path
        w.process_path.setPlainText(repr([cal, cal]))
        print("[TEST] two-spectrum path OK")
        QTimer.singleShot(200, self.step_build_model)

    # -- Step 4: build the model (synchronous) ------------------------

    def step_build_model(self) -> None:
        print("[TEST] Step: build model (Sextet, Be, Nbaseline, Sextet) ...")
        pt = self.window.params_table
        pt.select_model(1, "Sextet")        # 1: Sextet
        pt.select_model(2, "Doublet")       # 2: Doublet
        pt.select_model(3, "Nbaseline")     # 3: Nbaseline
        pt.select_model(2, "Delete")        # -> Sextet, Nbaseline
        pt.select_model(2, "Insert")        # -> Sextet, <insert>, Nbaseline
        pt.select_model(2, "Be")            # -> Sextet, Be(=Doublet), Nbaseline
        pt.select_model(4, "Sextet")        # -> Sextet, Be, Nbaseline, Sextet

        models = pt.get_model_list()
        if models != _EXPECTED_MODEL_LIST:
            self.errors.append(f"build model: got {models}, expected {_EXPECTED_MODEL_LIST}")
            self._finish()
            return
        print(f"[TEST] build model OK: {models}")
        QTimer.singleShot(300, self.step_show_model)

    # -- Step 5: show model (async) -----------------------------------

    def step_show_model(self) -> None:
        print("[TEST] Step: show model ...")
        self.window.showM_pressed()
        self._wait(self.step_fit, "show model")

    # -- Step 6: fit (async, simultaneous) ----------------------------

    def step_fit(self) -> None:
        print("[TEST] Step: fit ...")
        self.window.fit_pressed()
        self._wait(self.step_check_results, "fit")

    # -- Step 7: check chi^2 and colours (synchronous) ----------------

    def step_check_results(self) -> None:
        print("[TEST] Step: check results (chi^2 and colours) ...")
        w = self.window
        rt = w.results_table

        # (8) reduced chi-square
        chi2 = rt.current_chi2
        print(f"[TEST] chi^2 = {chi2}")
        if chi2 is None or not (chi2 < 2):
            self.errors.append(f"chi^2 check: chi2={chi2}, expected < 2")

        # (9) per-model colours must agree between the two tables.
        model_list = list(rt.current_model_list)
        n = len(model_list)
        param_colors = w.params_table.get_current_colors()      # [baseline, c1, c2, ...]
        result_stored = list(rt.current_model_colors)           # copied from the param table
        # Skip index 0 (baseline is rendered specially / has no colour button).
        param_model = list(param_colors[1:n])
        result_model = list(result_stored[1:n])
        # Colours actually rendered on the results-table buttons.
        rendered_model = [_parse_bg_color(rt.buttons[i * 3].styleSheet()) for i in range(1, n)]

        print(f"[TEST] param  model colours: {param_model}")
        print(f"[TEST] result model colours: {result_model}")
        print(f"[TEST] result rendered     : {rendered_model}")

        if param_model != result_model:
            self.errors.append(
                f"colour mismatch (param vs result-stored): {param_model} != {result_model}"
            )
        if rendered_model != param_model:
            self.errors.append(
                f"colour mismatch (param vs result-rendered): {param_model} != {rendered_model}"
            )
        if param_model != _EXPECTED_MODEL_COLORS:
            self.errors.append(
                f"colours {param_model} != expected {_EXPECTED_MODEL_COLORS}"
            )

        self._finish()

    # -- finish / close ------------------------------------------------

    def _finish(self) -> None:
        if self.errors:
            print("\n[TEST FAILED]")
            for msg in self.errors:
                print(f"  ERROR: {msg}")
        else:
            print("\n[TEST PASSED] No crashes, no errors in log.")
        QTimer.singleShot(200, self._close)

    def _close(self) -> None:
        self.window.close()
        self.app.quit()


def run_gui_smoke() -> int:
    """Run the GUI smoke test with a real multiprocessing pool.

    Entry point for ``syncmoss --test`` (works in source checkouts and inside the
    frozen binaries). Returns 0 on success, 1 on any failure. Protected parameter
    files and Calibration.dat are restored/left untouched afterwards.
    """
    mp.freeze_support()

    num_processes = mp.cpu_count() if mp.cpu_count() <= 4 else mp.cpu_count() - 1
    pool = mp.Pool(processes=max(1, num_processes))

    app = QApplication.instance() or QApplication(sys.argv)
    window = PhysicsApp(pool=pool)
    snapshot = _snapshot_param_files(window)
    _use_temp_calibration(window)
    window.show()

    runner = _TestRunner(app, window)
    QTimer.singleShot(500, runner.start)

    try:
        app.exec()
    finally:
        pool.close()
        pool.join()
        _restore_param_files(snapshot)

    return 1 if runner.errors else 0


# Backwards-compatible alias (the previous module exposed main()).
main = run_gui_smoke


def test_gui_open_and_fit(qapp, tmp_path):
    """pytest wrapper around the same smoke flow (head-less, ThreadPool).

    A ThreadPool replaces the multiprocessing pool so the test does not spawn
    subprocesses (fragile under pytest); the app only uses ``pool.starmap``, which
    ThreadPool provides with identical semantics, so the same fitting code runs.
    """
    pool = ThreadPool(processes=2)
    window = PhysicsApp(pool=pool)
    snapshot = _snapshot_param_files(window)
    # Operate on a temp copy so the fit never rewrites the tracked Calibration.dat.
    if os.path.exists(window.calibration_path):
        tmp_copy = os.path.join(str(tmp_path), "Calibration.dat")
        shutil.copy2(window.calibration_path, tmp_copy)
        window.calibration_path = tmp_copy
    try:
        window.show()
        runner = _TestRunner(qapp, window)

        def _safety_quit():
            if window.inprogress and not runner.errors:
                runner.errors.append("safety timeout: GUI smoke test did not finish")
            qapp.quit()

        QTimer.singleShot(300_000, _safety_quit)
        QTimer.singleShot(300, runner.start)
        qapp.exec()

        assert runner.errors == [], "GUI smoke test failed: " + " | ".join(runner.errors)
    finally:
        window.close()
        pool.close()
        pool.join()
        _restore_param_files(snapshot)


if __name__ == "__main__":
    sys.exit(run_gui_smoke())
