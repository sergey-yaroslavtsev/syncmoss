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
  4. switch to a TWO-spectrum path ([Calibration.dat, Calibration.dat]) and show it
  5. build the model: add Sextet, Doublet, Nbaseline; delete the Doublet; Insert a
     row before Nbaseline; turn that row into a Be model; add a Sextet after
     Nbaseline  ->  Sextet, Be(=Doublet), Nbaseline, Sextet
  6. show the model
  7. fit (simultaneous, because of Nbaseline + two spectra)
  8. assert reduced chi-square < 3
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

# Upper bound on the final reduced chi-square (see step_check_results).
_MAX_CHI2 = 3.0

# Expected final model component colours (baseline excluded). The user verified
# these are produced by the delete/insert/Be sequence below; the key invariant is
# that the parameters table and the results table agree on them.
_EXPECTED_MODEL_COLORS = ['red', 'cyan', 'cyan', 'yellow']
# get_model_list() reports a Be model as a "Doublet" (it is a Doublet preset).
_EXPECTED_MODEL_LIST = ['baseline', 'Sextet', 'Doublet', 'Nbaseline', 'Sextet']


def _parse_bg_color(stylesheet: str):
    """Extract the ``background-color`` value from a Qt stylesheet string."""
    match = re.search(r'background-color:\s*([^;]+);', stylesheet or '')
    return match.group(1).strip() if match else None


def _redirect_params_to_tmp(window, tmp_dir=None) -> None:
    """Point the window's parameter directory at a throw-away copy.

    The smoke flow both shows/fits the bundled Calibration.dat AND refines the
    instrumental function (step 2), which rewrites ``INSexp.txt`` / ``INSint.txt``
    in ``params_dir`` (and re-calibrating rewrites Calibration.dat in place). Left
    on the real ``parameters/`` folder, those tracked data files would be mutated
    and show up as spurious modifications on every commit. Copying the whole
    parameters folder into a temp dir and repointing ``params_dir`` +
    ``calibration_path`` there keeps the repo clean while all the parameter files
    the fit reads (Be.txt, GCMS.txt, ...) remain available.

    ``tmp_dir`` is used when given (e.g. pytest's ``tmp_path``, auto-cleaned);
    otherwise a fresh temp dir is created (``--test`` in source / frozen runs).
    """
    src = window.params_dir
    if not src or not os.path.isdir(src):
        return
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="syncmoss_smoke_")
    dst = os.path.join(str(tmp_dir), "parameters")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    window.params_dir = dst
    window.calibration_path = os.path.join(dst, "Calibration.dat")


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
        # Show them, as a user does (choosing files through the dialog has the
        # same effect): Nbaseline takes its starting Ns from the background of
        # the matching entry of path_list, and without this path_list still
        # holds the single spectrum of step 1, so Ns fell back to 10000 against
        # ~1.4e5 counts and the fit could land in a wrong minimum.
        w.show_pressed()  # synchronous
        if "red" in w.log.styleSheet():
            self.errors.append(f"set two-spectrum path: {w.log.toPlainText()}")
            self._finish()
            return
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

        # (8) reduced chi-square. The bar is set by the bundled Calibration.dat:
        # at ~1.4e5 counts per channel the instrumental refinement of step 2
        # already ends at 2.43, and this fit converges to 2.80 (the second
        # spectrum carries no Be doublet). A fit stuck in a wrong minimum is
        # ~100x that. The old 6e3-count calibration passed at < 2.
        chi2 = rt.current_chi2
        print(f"[TEST] chi^2 = {chi2}")
        if chi2 is None or not (chi2 < _MAX_CHI2):
            self.errors.append(f"chi^2 check: chi2={chi2}, expected < {_MAX_CHI2}")

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
    _redirect_params_to_tmp(window)
    window.show()

    runner = _TestRunner(app, window)
    QTimer.singleShot(500, runner.start)

    try:
        app.exec()
    finally:
        pool.close()
        pool.join()

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
    # Operate on a temp copy of the whole parameters folder so neither the fit
    # nor the instrumental refinement (step 2) rewrites the tracked data files
    # (Calibration.dat, INSexp.txt, INSint.txt).
    _redirect_params_to_tmp(window, tmp_path)
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


if __name__ == "__main__":
    sys.exit(run_gui_smoke())
