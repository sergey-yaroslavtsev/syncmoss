"""Stand-ins for compile_functions.compile_functions, run by the tests in the
new process of jit_cache.run_in_new_process. A module of their own, importing
nothing heavy: that process imports it to find them."""
import os
import time

# Where leave_an_index leaves its index (the tests set it; the process inherits it)
FOLDER_VARIABLE = 'SYNCMOSS_TEST_NUMBA_CACHE'


def index(folder, name="models.Voigt-381.py313"):
    """What numba leaves for one function: its index and one compiled version.
    Returns the index."""
    sub = os.path.join(str(folder), "syncmoss")
    os.makedirs(sub, exist_ok=True)
    for ending in (".nbi", ".1.nbc"):
        with open(os.path.join(sub, name + ending), "wb") as f:
            f.write(b"numba")
    return os.path.join(sub, name + ".nbi")


def leave_an_index(report):
    report(0, 2, "probe")
    report(1, 2, "probe again")
    index(os.environ[FOLDER_VARIABLE])


def fail(report):
    raise RuntimeError("no compiler today")


def sleep_long(report):
    time.sleep(600)
