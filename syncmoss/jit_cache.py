"""numba's compiled functions: where they are kept, whether they are complete
for the running program, and compiling them in a process of its own.

Every calculation function of SYNCmoss is compiled by numba (``@njit(cache=True)``)
the first time it runs, and the machine code is kept on disk for the next starts
and for the pool's processes. The folder is the program's own, ``numba_cache``
next to its ``parameters/`` folder (``cache_dir``):

* the Windows bundle: next to ``SYNCmoss.exe``;
* the frozen macOS app: ``~/Library/Application Support/SYNCmoss``, with its
  parameters (app_paths.macos_app_data_dir);
* a pip install or a source checkout: the package folder -- for a pip install
  inside its environment (site-packages), like the parameters.

numba takes it from ``NUMBA_CACHE_DIR`` and fixes each function's folder when the
function is decorated, i.e. when its module is imported: so ``configure`` runs at
the top of main.py, before anything imports numba, and the processes the program
starts inherit the variable. A frozen app's modules have no source file, which
numba's own locator for ``NUMBA_CACHE_DIR`` requires: there
frozen_cache_locator.FrozenCacheLocator comes first.

A start compiles them all (compile_window, compile_functions) unless the folder
holds a finished compiling for the running program: MARKER, written when one
finishes, says what it was for (``stamp``). Between starts numba decides by
itself: a function whose source file changed (from source) is compiled again at
its first use.
"""
import glob
import json
import multiprocessing as mp
import os
import sys
import traceback

from syncmoss.app_paths import app_dir, is_frozen_macos, macos_app_data_dir

FOLDER_NAME = 'numba_cache'
# Written when a compiling finished: what it was for (stamp)
MARKER = 'compiled.json'
# numba's files: an index per function, the machine code of each compiled version
_INDEX = '*.nbi'
_FILES = (_INDEX, '*.nbc')
# NUMBA_CACHE_LOCATOR_CLASSES of a frozen app: the folder, else numba's own
# per-user one (a name without a module is one of numba's)
FROZEN_LOCATORS = 'syncmoss.frozen_cache_locator.FrozenCacheLocator,UserWideCacheLocator'


def cache_dir():
    """The folder of the compiled functions: ``numba_cache`` next to the
    parameters folder (see the module's notes)."""
    if is_frozen_macos():
        return os.path.join(macos_app_data_dir(), FOLDER_NAME)
    return os.path.join(app_dir(), FOLDER_NAME)


def configure():
    """Point numba at cache_dir(), in this process and the ones it starts.

    Only where numba is not imported yet: once it is, the functions already
    decorated keep their folder and only later ones would move. That is a
    process which is not the program's start (a test importing main.py for its
    version), and it is left as it is."""
    if 'numba' in sys.modules:
        return
    os.environ['NUMBA_CACHE_DIR'] = cache_dir()
    if getattr(sys, 'frozen', False):
        os.environ['NUMBA_CACHE_LOCATOR_CLASSES'] = FROZEN_LOCATORS


def _numba_files(folder):
    return [path for pattern in _FILES
            for path in glob.glob(os.path.join(glob.escape(folder), '**', pattern), recursive=True)]


def compiled(folder):
    """Whether *folder* holds a compiled function (an index of numba's)."""
    return bool(glob.glob(os.path.join(glob.escape(folder), '**', _INDEX), recursive=True))


def files(folder):
    """numba's files in *folder*, each with its time stamp and size: what a
    compiling added or rewrote shows as a difference."""
    found = {}
    for path in _numba_files(folder):
        st = os.stat(path)
        found[os.path.relpath(path, folder)] = (st.st_mtime_ns, st.st_size)
    return found


def clear(folder):
    """Delete numba's files and MARKER in *folder* (nothing else), so that
    every function is compiled anew. One that cannot be deleted is overwritten
    by the next compiling."""
    for path in _numba_files(folder) + [os.path.join(folder, MARKER)]:
        try:
            os.remove(path)
        except OSError:
            pass


def stamp(version, numba_version):
    """What a compiling is for: SYNCmoss *version*, *numba_version* (numba
    keeps nothing of another one) and, in a frozen app, the executable's time
    stamp and size -- what numba dates the functions by there, so a new build
    compiles anew."""
    marked = {'version': version, 'numba': numba_version}
    if getattr(sys, 'frozen', False):
        executable = os.stat(sys.executable)
        marked['executable'] = [executable.st_mtime, executable.st_size]
    return marked


def why_compile(folder, current):
    """Why *folder* is to be compiled for the stamp *current*, or None when it
    holds a finished compiling for it."""
    try:
        with open(os.path.join(folder, MARKER), encoding='utf-8') as f:
            marked = json.load(f)
    except FileNotFoundError:
        return "the last compiling did not finish" if compiled(folder) else "nothing compiled yet"
    except (OSError, ValueError):
        marked = None
    if not isinstance(marked, dict):
        return f"{MARKER} unreadable"
    for key, name in (('version', 'SYNCmoss'), ('numba', 'numba')):
        if marked.get(key) != current[key]:
            return f"compiled for {name} {marked.get(key)}, this is {current[key]}"
    if marked.get('executable') != current.get('executable'):
        return "compiled by another executable"
    if not compiled(folder):
        return "no compiled function left"
    return None


def mark_compiled(folder, current):
    """Record in *folder* that a compiling for *current* finished."""
    with open(os.path.join(folder, MARKER), 'w', encoding='utf-8') as f:
        json.dump(current, f)


class CompilingFailed(Exception):
    """The compiling raised (its traceback), or its process ended without a word."""


def in_new_process(connection, work):
    """The new process's body: *work* with a reporter of its steps, each sent
    through *connection* as ('step', done, steps, name); then ('done', None),
    or ('done', the traceback of its error)."""
    def report(done, steps, name):
        connection.send(('step', done, steps, name))
    try:
        work(report)
    except BaseException:
        connection.send(('done', traceback.format_exc()))
    else:
        connection.send(('done', None))
    finally:
        connection.close()


def run_in_new_process(work, progress=None, stop=None):
    """*work* (compile_functions.compile_functions) in a new process, waited
    for here: 'spawn', an interpreter of its own on every system, holding no
    compiled function (numba neither compiles nor stores again what its
    process holds). A daemon: the program's end ends it too.

    *progress* gets the steps the work reports; *stop*, asked every 0.1 s,
    terminates it when it says True. True when the work finished, False when
    stopped; CompilingFailed when it failed."""
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=in_new_process, args=(sender, work), daemon=True)
    process.start()
    sender.close()                              # the process writes through its own copy
    outcome = None
    try:
        while outcome is None:
            if receiver.poll(0.1):              # True as well once the process is gone
                message = receiver.recv()
                if message[0] == 'step':
                    if progress is not None:
                        progress(*message[1:])
                else:
                    outcome = ('done', message[1])
            elif stop is not None and stop():
                process.terminate()
                outcome = ('stopped', None)
    except (EOFError, OSError):                 # gone without a word: the pipe broke
        outcome = ('lost', None)
    process.join()
    receiver.close()
    if outcome[0] == 'stopped':
        return False
    if outcome == ('done', None):
        return True
    raise CompilingFailed(outcome[1] or f"the compiling process ended with exit code {process.exitcode}")
