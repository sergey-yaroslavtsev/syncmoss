"""The compiling of the calculation functions at a start that needs it: the
first start, a new version, a compiling that did not finish (jit_cache's
why_compile). main.py only, like supp_menu.apply_saved_settings: the tests'
windows do not compile.

compile_functions runs in a process of its own (jit_cache.run_in_new_process)
while a small window says so and counts the steps. The main window can be used
meanwhile, its calculations wait (inprogress) and Interrupt stops the
compiling; whatever stopped it, the next start compiles again.
"""
import numba
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QProgressBar, QVBoxLayout

import syncmoss.jit_cache as jit_cache
from syncmoss.compile_functions import compile_functions
from syncmoss.error_reporter import app_version
from syncmoss.models import FIT_CANCEL, FitInterrupted

# What the window is busy with: "Fit is not available right now: <BUSY> is still running"
BUSY = "Compiling the functions"


def current_stamp():
    """What a compiling for the running program is marked with (jit_cache.stamp)."""
    return jit_cache.stamp(app_version(), numba.__version__)


def compile_if_needed(main_window, work=compile_functions, fresh=False):
    """At the start: when the cache folder holds no finished compiling for
    the running program -- or whatever it holds, *fresh* (main.py's --fresh) --
    empty it and compile everything (*work*) in a new process. Returns True
    when it started."""
    folder = main_window.numba_cache_dir
    current = current_stamp()
    reason = "asked for (--fresh)" if fresh else jit_cache.why_compile(folder, current)
    if reason is None:
        return False
    print(f"[Compile] {folder}: {reason}; compiling the calculation functions")
    jit_cache.clear(folder)
    main_window.inprogress = True
    main_window.busy_with = BUSY
    main_window.set_status("Compiling the calculation functions (about a minute)...", "cyan")
    dialog = main_window.compile_dialog = CompileDialog(main_window, folder, current)
    thread = main_window.compile_thread = CompileThread(work)
    # methods of the window, so that they run in its thread
    thread.progress.connect(dialog.show_step)
    thread.done.connect(dialog.finish)
    QApplication.instance().aboutToQuit.connect(thread.stop)
    dialog.show()
    main_window._start_calculation(thread)
    return True


class CompileDialog(QDialog):
    """The small window of the compiling: what it does and how far it is. It
    closes itself at the end (finish); closed before, it only hides."""

    def __init__(self, main_window, folder, current):
        super().__init__(main_window)
        self.main_window, self.folder, self.current = main_window, folder, current
        self.setWindowTitle("Compiling")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Compiling the calculation functions for this version of SYNCmoss:\n"
                                "about a minute, once. Calculations wait until it is done.", self))
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)                 # busy until the first step is reported
        layout.addWidget(self.bar)

    def show_step(self, done, steps, name):
        self.bar.setRange(0, steps)
        self.bar.setValue(done)
        self.bar.setFormat(f"{name}  %v/%m")

    def finish(self, error):
        """The compiling ended (CompileThread.done): marked in the folder when
        it finished, else said why not; the window closes, the main window is
        released."""
        w = self.main_window
        QApplication.instance().aboutToQuit.disconnect(w.compile_thread.stop)
        self.close()
        if error is not None:
            w.set_status(f"Compiling the functions failed: {error}. Each model is compiled at its first "
                         "calculation; the next start tries again.", "red")
        else:
            try:
                jit_cache.mark_compiled(self.folder, self.current)
                w.set_status("Calculation functions compiled: ready", "green")
            except OSError as e:
                w.set_status(f"Calculation functions compiled, but they cannot be marked so in {self.folder} "
                             f"({e}): the next start compiles again.", "orange")
        w._calculation_ended()


class CompileThread(QThread):
    """jit_cache.run_in_new_process with *work*, waited for here. Interrupt
    (models.FIT_CANCEL) and the program's end (stop) terminate the process.

    Its signals: 'progress', the steps the work reports, and 'done' -- not
    'finished', which would shadow QThread.finished: None, or why the
    compiling did not finish. Kept on the main window (compile_thread), so
    Interrupt waits for it.
    """
    progress = Signal(int, int, str)
    done = Signal(object)

    def __init__(self, work):
        super().__init__()
        self.work = work
        self._stopping = False

    def run(self):
        try:
            finished = jit_cache.run_in_new_process(self.work, self.progress.emit,
                                                    lambda: self._stopping or FIT_CANCEL.is_set())
        except jit_cache.CompilingFailed as e:
            print(f"[ERROR] Compiling the functions failed: {e}")
            self.done.emit(str(e).strip().splitlines()[-1])
        else:
            self.done.emit(None if finished else str(FitInterrupted()))

    def stop(self):
        """At the program's end: terminate the process and wait, so that the
        thread does not outlive the program."""
        self._stopping = True
        self.wait()
