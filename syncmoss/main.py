import os
import sys
import multiprocessing as mp
from PySide6.QtWidgets import QApplication
from syncmoss.syncmoss_main import PhysicsApp
__VERSION__ = "0.2.0"


def _run_gui_smoke():
    """Run the GUI smoke test (used by ``syncmoss --test``).

    The smoke driver lives in ``tests/syncmoss_test.py`` (so it is the single
    source of truth shared with the pytest suite). It is bundled into the frozen
    binaries via the PyInstaller specs, where it imports as the top-level module
    ``syncmoss_test``. In a source checkout we add the repo ``tests/`` dir to the
    import path first.
    """
    try:
        import syncmoss_test
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))      # .../syncmoss/syncmoss
        tests_dir = os.path.join(os.path.dirname(here), 'tests')
        if os.path.isdir(tests_dir) and tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        import syncmoss_test
    return syncmoss_test.run_gui_smoke()


def main():
    mp.freeze_support()

    if '--test' in sys.argv:
        sys.argv.remove('--test')
        return _run_gui_smoke()

    # Create global pool once at startup (reused throughout application)
    # Use same logic as original: all cores if <=4, else cores-1
    num_processes = mp.cpu_count() if mp.cpu_count() <= 4 else mp.cpu_count() - 1
    global pool
    pool = mp.Pool(processes=num_processes)

    app = QApplication(sys.argv)
    window = PhysicsApp(pool=pool)
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
