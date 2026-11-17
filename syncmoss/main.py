import io
import os
import sys
import time
import multiprocessing as mp

# Where numba keeps the compiled functions: set before the imports below load
# numba, in this process and (inherited) in every one it starts
import syncmoss.jit_cache as jit_cache
jit_cache.configure()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from syncmoss import console_io
import syncmoss.error_reporter as error_reporter
from syncmoss.syncmoss_main import PhysicsApp, new_pool
from syncmoss.supp_menu import apply_saved_settings
from syncmoss.compile_functions import compile_functions
from syncmoss.compile_window import compile_if_needed, current_stamp
__VERSION__ = "0.8.2"

# What --test leaves in the cache folder for --test-again: the seconds of its
# plain fit before anything was compiled
_FIRST_FIT_FILE = 'test_first_fit.txt'
# --test-again: its fit must take at most this part of that one
_SECOND_FIT_SHARE = 0.5


def _smoke():
    """The smoke driver, ``tests/syncmoss_test.py`` (the single source of truth
    shared with the pytest suite). It is bundled into the frozen binaries via
    the PyInstaller specs, where it imports as the top-level module
    ``syncmoss_test``; in a source checkout the repo ``tests/`` dir is added to
    the import path first; for a pip install it is put on PYTHONPATH (CI).
    """
    try:
        import syncmoss_test
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))      # .../syncmoss/syncmoss
        tests_dir = os.path.join(os.path.dirname(here), 'tests')
        if os.path.isdir(tests_dir) and tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        import syncmoss_test
    return syncmoss_test


def _compile_all(folder):
    """Every calculation function compiled anew into *folder* as a start
    compiles them (a new process: jit_cache.run_in_new_process), its steps
    printed, and marked for this program. None, or why it failed (in a frozen
    app the folder must be frozen_cache_locator's)."""
    jit_cache.clear(folder)
    try:
        jit_cache.run_in_new_process(
            compile_functions, progress=lambda done, steps, name: print(f"[Compile] {done}/{steps} {name}", flush=True))
    except jit_cache.CompilingFailed as e:
        return f"compiling the functions: {e}"
    if not jit_cache.compiled(folder):
        return f"the compiling left nothing in {folder}"
    jit_cache.mark_compiled(folder, current_stamp())
    return None


def _run_compile():
    """``syncmoss --compile``: what --fresh does at a start, without the GUI --
    every calculation function compiled anew, e.g. after the calculation code
    was edited."""
    folder = jit_cache.cache_dir()
    started = time.perf_counter()
    error = _compile_all(folder)
    if error is not None:
        print(f"FAILED: {error}")
        return 1
    print(f"[Compile] compiled in {time.perf_counter() - started:.0f} s into {folder}")
    return 0


def _run_test():
    """``syncmoss --test``, a first start: with the cache folder emptied, a
    plain fit timed (what a session's first fit cost before anything was
    compiled), then the calculation functions compiled anew as a start compiles
    them (_compile_all), then the GUI smoke flow, which must find every function
    it runs compiled -- nothing added to the folder. The fit's time stays in the
    folder for --test-again."""
    smoke = _smoke()
    folder = jit_cache.cache_dir()
    jit_cache.clear(folder)
    first_fit = smoke.timed_fit()
    print(f"[Test] a plain fit, nothing compiled: {first_fit:.1f} s", flush=True)
    error = _compile_all(folder)             # from an empty folder again, as a first start's
    if error is not None:
        print(f"FAILED: {error}")
        return 1
    with open(os.path.join(folder, _FIRST_FIT_FILE), 'w', encoding='utf-8') as f:
        f.write(repr(first_fit))
    compiled = jit_cache.files(folder)
    print(f"[Compile] {len(compiled)} files in {folder}", flush=True)
    failed = smoke.run_gui_smoke()
    added = sorted(name for name, state in jit_cache.files(folder).items() if compiled.get(name) != state)
    if added:
        print(f"FAILED: the smoke flow compiled what the compiling had not: {', '.join(added)}")
        return 1
    return failed


def _run_test_again():
    """``syncmoss --test-again``, a second start after --test: the cache folder
    must hold the finished compiling for this program (this start compiles
    nothing), and the same plain fit must add nothing to it and take at most
    _SECOND_FIT_SHARE of the time it took there, before anything was compiled."""
    folder = jit_cache.cache_dir()
    reason = jit_cache.why_compile(folder, current_stamp())
    if reason is not None:
        print(f"FAILED: a second start would compile again: {reason}")
        return 1
    try:
        with open(os.path.join(folder, _FIRST_FIT_FILE), encoding='utf-8') as f:
            first_fit = float(f.read())
    except (OSError, ValueError):
        print(f"FAILED: no time of --test's fit in {folder}: run --test first")
        return 1
    compiled = jit_cache.files(folder)
    second_fit = _smoke().timed_fit()
    print(f"[Test] the same fit at a second start: {second_fit:.1f} s "
          f"(before anything was compiled: {first_fit:.1f} s, {first_fit / second_fit:.1f} times longer)", flush=True)
    added = sorted(name for name, state in jit_cache.files(folder).items() if compiled.get(name) != state)
    if added:
        print(f"FAILED: the second start compiled {', '.join(added)}")
        return 1
    if second_fit > _SECOND_FIT_SHARE * first_fit:
        print(f"FAILED: the fit at the second start is not faster than {_SECOND_FIT_SHARE} of the first")
        return 1
    print("[TEST PASSED] The second start found everything compiled.")
    return 0


def main():
    # A process of the pool or of the compiling runs its work in here and ends
    mp.freeze_support()
    # The windowed Windows build: the terminal it was started from, if any, or
    # with --console a console window of its own (console_io); our switches
    # are taken out of sys.argv, so that Qt does not see them
    console_io.enable_console(force=console_io.consume_flag(sys.argv, '--console'))
    # print() writes UTF-8. Where stdout is a pipe or a file on Windows (CI,
    # `SYNCmoss.exe --test > log`) Python takes the ANSI code page, which has
    # no χ: printing a fit result raised UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')

    # --fresh: this start compiles the calculation functions anew, whatever the
    # folder holds -- e.g. after the calculation code was edited: numba dates
    # a function by its own file only, not by the files of the functions it calls
    fresh = console_io.consume_flag(sys.argv, '--fresh')

    if console_io.consume_flag(sys.argv, '--compile'):
        return _run_compile()
    if console_io.consume_flag(sys.argv, '--test'):
        return _run_test()
    if console_io.consume_flag(sys.argv, '--test-again'):
        return _run_test_again()

    # Create global pool once at startup (reused throughout application):
    # all cores if <=4, else cores-1, each process's BLAS in one thread (new_pool)
    global pool
    pool = new_pool()

    app = QApplication(sys.argv)
    app.setApplicationName("SYNCmoss")

    # From here on every traceback that reaches the terminal — uncaught, or
    # caught-and-printed somewhere in the app — also raises the "report a bug"
    # window. Installed after QApplication (it needs the GUI thread's event
    # loop) and before the main window, so a crash while building it is covered.
    error_reporter.install()

    try:
        window = PhysicsApp(pool=pool)
        # Supp -> "Save settings as default"; only here, so a window built by
        # the tests or --test has the built-in settings
        apply_saved_settings(window)
        window.show()
        # The first start of this version (or an unfinished compiling): the
        # calculation functions compiled, its small window over the main one
        QTimer.singleShot(0, lambda: compile_if_needed(window, fresh=fresh))
    except Exception:
        # No event loop yet, so the normal deferred window would never open.
        error_reporter.report_exception(blocking=True)
        return 1

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
