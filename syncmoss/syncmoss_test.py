"""
Automated smoke test for SYNCmoss GUI.

Run from c:\\Users\\yaroslav\\PycharmProjects\\SYNCmoss\\syncmoss\\ :
    python syncmoss_test.py

Steps:
1. Start GUI
2. Add Sextet model to row 1
3. Show spectrum  (Calibration.dat used as the test spectrum)
4. Show model
5. Fit
6. Verify no crash and no errors in the log widget
7. Close GUI

Exit code: 0 on success, 1 on any failure.
"""

import sys
import os
import multiprocessing as mp

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from syncmoss.syncmoss_main import PhysicsApp


# How often to re-check whether an async operation has finished (ms)
_POLL_MS = 500
# Max polls before giving up (~50 s at 500 ms each)
_TIMEOUT_POLLS = 100


class _TestRunner:
    """Drives the GUI through the smoke-test steps sequentially."""

    def __init__(self, app: QApplication, window: PhysicsApp) -> None:
        self.app = app
        self.window = window
        self.errors: list = []
        self._poll_count = 0

    # ------------------------------------------------------------------
    # Step 1: add Sextet model
    # ------------------------------------------------------------------

    def step_add_sextet(self) -> None:
        print("[TEST] Step 1: Adding Sextet model to row 1 ...")
        w = self.window
        w.params_table.select_model(1, "Sextet")

        # Verify the model button text was updated
        row_widget = w.params_table.row_widgets[1]
        model_btn = (
            row_widget.layout().itemAt(0).widget().layout().itemAt(1).widget()
        )
        actual = model_btn.text()
        if actual != "Sextet":
            self.errors.append(
                f"Step 1: model button shows '{actual}', expected 'Sextet'"
            )
        else:
            print("[TEST] Step 1 OK")

        QTimer.singleShot(300, self.step_show_spectrum)

    # ------------------------------------------------------------------
    # Step 2: show spectrum  (synchronous)
    # ------------------------------------------------------------------

    def step_show_spectrum(self) -> None:
        print("[TEST] Step 2: Showing spectrum (Calibration.dat) ...")
        w = self.window

        cal_path = w.calibration_path          # absolute path set during __init__
        if not os.path.exists(cal_path):
            self.errors.append(
                f"Step 2: Calibration.dat not found at '{cal_path}'"
            )
            self._finish()
            return

        # Set the spectrum path in the text field and trigger show
        w.process_path.setPlainText(repr([cal_path]))
        w.show_pressed()                       # synchronous; inprogress is cleared on return

        if "red" in w.log.styleSheet():
            self.errors.append(f"Step 2: {w.log.toPlainText()}")
        else:
            print("[TEST] Step 2 OK")

        QTimer.singleShot(300, self.step_show_model)

    # ------------------------------------------------------------------
    # Step 3: show model  (async – uses ShowModelThread)
    # ------------------------------------------------------------------

    def step_show_model(self) -> None:
        print("[TEST] Step 3: Showing model (async) ...")
        self.window.showM_pressed()
        self._poll_count = 0
        QTimer.singleShot(_POLL_MS, self._wait_show_model)

    def _wait_show_model(self) -> None:
        if not self.window.inprogress:
            if "red" in self.window.log.styleSheet():
                self.errors.append(f"Step 3: {self.window.log.toPlainText()}")
            else:
                print("[TEST] Step 3 OK")
            QTimer.singleShot(300, self.step_fit)
        elif self._poll_count >= _TIMEOUT_POLLS:
            self.errors.append("Step 3: timeout waiting for show model to finish")
            self._finish()
        else:
            self._poll_count += 1
            QTimer.singleShot(_POLL_MS, self._wait_show_model)

    # ------------------------------------------------------------------
    # Step 4: fit  (async – uses FittingThread)
    # ------------------------------------------------------------------

    def step_fit(self) -> None:
        print("[TEST] Step 4: Fitting (async) ...")
        self.window.fit_pressed()
        self._poll_count = 0
        QTimer.singleShot(_POLL_MS, self._wait_fit)

    def _wait_fit(self) -> None:
        if not self.window.inprogress:
            if "red" in self.window.log.styleSheet():
                self.errors.append(f"Step 4: {self.window.log.toPlainText()}")
            else:
                print("[TEST] Step 4 OK")
            self._finish()
        elif self._poll_count >= _TIMEOUT_POLLS:
            self.errors.append("Step 4: timeout waiting for fit to finish")
            self._finish()
        else:
            self._poll_count += 1
            QTimer.singleShot(_POLL_MS, self._wait_fit)

    # ------------------------------------------------------------------
    # Finish: report and close
    # ------------------------------------------------------------------

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


def main() -> int:
    mp.freeze_support()

    num_processes = mp.cpu_count() if mp.cpu_count() <= 4 else mp.cpu_count() - 1
    pool = mp.Pool(processes=num_processes)

    app = QApplication(sys.argv)
    window = PhysicsApp(pool=pool)
    window.show()

    runner = _TestRunner(app, window)
    # Give the window time to fully render before the first step
    QTimer.singleShot(500, runner.step_add_sextet)

    app.exec()

    pool.close()
    pool.join()

    return 1 if runner.errors else 0


if __name__ == "__main__":
    sys.exit(main())
