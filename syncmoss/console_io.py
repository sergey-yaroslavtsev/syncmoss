# -*- coding: utf-8 -*-
"""console_io.py -- give the windowed Windows build a working stdout.

The Windows executable is built for the GUI subsystem (``console=False``), so
double-clicking it never flashes a console window. The price is that such a
process starts with NO console at all: file descriptors 0/1/2 are invalid and
``sys.stdout`` / ``sys.stderr`` are ``None`` -- what it prints is lost, and code
writing to them directly fails. That is why the bundle used to ship a SECOND
executable (``SYNCmoss_console.exe``) whose only purpose was to be visible in a
terminal. (Ported from the SYNCtime branch, where it replaced that one too.)

One executable is enough. main.py calls :func:`enable_console` before anything
prints (after multiprocessing's freeze_support: the pool's processes stay
silent):

  ========================================== ===================================
  how the .exe was started                   what happens
  ========================================== ===================================
  double-click (no parent console)           nothing: stays silent, and no
                                             console window ever appears
  from PowerShell / cmd                      ``AttachConsole(parent)`` -- the
                                             prints land in that terminal
  redirected or piped                        the inherited handle is already
  (``> log.txt``, ``| Out-Host``)            valid and is used untouched
  ``--console`` with no console to attach    ``AllocConsole()`` -- a console
  to (i.e. double-clicked)                   window of our own, held open at
                                             exit so a traceback stays readable
  ========================================== ===================================

Everything here is a no-op on Linux and macOS, where a GUI binary simply keeps
the stdio it inherited.

One Windows wart is not fixable from inside the process: a GUI-subsystem
executable does not make the calling shell wait for it, so the prompt returns
immediately and the output arrives after it. Pipe the exe when that matters --
``SYNCmoss.exe --test | Out-Host`` -- PowerShell then waits and sets
``$LASTEXITCODE``; ``Start-Process -Wait -PassThru`` is the explicit form, and
is what the CI smoke steps use.
"""

import os
import sys

#: ``AttachConsole`` parameter: attach to the console of the parent process.
ATTACH_PARENT_PROCESS = -1


def enable_console(force=False):
    """Make ``print`` work in a frozen windowed build. Returns True if it does.

    *force* (the ``--console`` flag) additionally allocates a console window
    when there is no parent console to attach to, which is the only way to see
    output from an executable started by double-clicking it.
    """
    if sys.platform != 'win32':
        return True                     # POSIX GUI binaries keep their stdio

    if _fd_is_valid(1):
        # Output is redirected or piped, or this is a console process: that
        # handle is exactly what the caller asked for, so leave it alone. A
        # windowed build can still have sys.stdout == None on top of it.
        if sys.stdout is None or sys.stderr is None:
            _rebind_streams()
        return True

    import ctypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    attached = bool(kernel32.AttachConsole(ATTACH_PARENT_PROCESS))
    allocated = False
    if not attached and force:
        allocated = bool(kernel32.AllocConsole())
    if not (attached or allocated):
        return False                    # double-clicked: nothing to print to

    if not _open_console_descriptors():
        return False
    _rebind_streams()
    if allocated:
        _hold_console_open()
    return True


def consume_flag(argv, *names):
    """Remove *names* from *argv* (in place) and report whether any was there.

    Case-insensitive. main.py takes its own switches out this way, so that
    neither Qt nor the application sees them.
    """
    wanted = {str(name).lower() for name in names}
    found = False
    for index in range(len(argv) - 1, -1, -1):
        if str(argv[index]).lower() in wanted:
            del argv[index]
            found = True
    return found


def _fd_is_valid(fd):
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


def _open_console_descriptors():
    """Point descriptors 0/1/2 at the console we just attached to.

    Rebinding ``sys.stdout`` alone would be enough for this process, but the
    descriptors are what native code (numba, Qt) writes to.
    """
    try:
        fd = os.open('CONOUT$', os.O_WRONLY | getattr(os, 'O_BINARY', 0))
    except OSError:
        return False
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)

    try:                                # stdin is optional; nothing reads it
        fd = os.open('CONIN$', os.O_RDONLY | getattr(os, 'O_BINARY', 0))
    except OSError:
        return True
    try:
        os.dup2(fd, 0)
    finally:
        os.close(fd)
    return True


def _rebind_streams():
    """(Re)build sys.stdin/stdout/stderr on top of descriptors 0/1/2."""
    if _fd_is_valid(0):
        sys.stdin = _wrap(0, 'r')
        if sys.__stdin__ is None:
            sys.__stdin__ = sys.stdin
    for fd, name in ((1, 'stdout'), (2, 'stderr')):
        if not _fd_is_valid(fd):
            continue
        stream = _wrap(fd, 'w')
        setattr(sys, name, stream)
        if getattr(sys, '__%s__' % name) is None:
            setattr(sys, '__%s__' % name, stream)


def _wrap(fd, mode):
    """A text stream on *fd*, line-buffered for writing.

    The builtin ``open`` (rather than ``io.FileIO``) is deliberate: on Windows
    it detects a console descriptor and uses CPython's ``_WindowsConsoleIO``,
    which writes through ``WriteConsoleW``. That is what keeps non-ASCII output
    ('Moessbauer', 'delta') legible whatever the console code page is.
    """
    return open(fd, mode, buffering=1 if mode == 'w' else -1,
                encoding='utf-8', errors='replace', closefd=False)


def _hold_console_open():
    """Keep a console WE allocated alive until the user has read it.

    Such a console is owned by this process and disappears with it, which would
    wipe the traceback the ``--console`` flag was used to see.
    """
    import atexit

    def wait_for_enter():
        try:
            print()
            print('--- press Enter to close this console ---')
            sys.stdin.readline()
        except Exception:                                     # noqa: BLE001
            pass                        # never let the exit path raise

    atexit.register(wait_for_enter)
