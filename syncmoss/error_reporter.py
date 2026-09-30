"""Global error capture and the "Report a bug" window.

No other module has to cooperate. The reporter is installed once (from
:func:`syncmoss.main.main`) and picks errors up from a single, global place:
a *tee* on ``sys.stdout`` and ``sys.stderr``. That covers both kinds of error
this application produces —

* the ones nobody catches, which Python's default ``sys.excepthook`` /
  ``threading.excepthook`` / ``sys.unraisablehook`` print to ``sys.stderr``
  (all three go through the Python-level stream object, so replacing it is
  enough — the hooks themselves are left untouched), and
* the ones the app catches itself and turns into a red status line after a
  ``print(traceback.format_exc())`` — hundreds of call sites, which is exactly
  why this is done globally instead of per module.

The tee **writes through first and unconditionally**, so every message still
reaches the terminal, in the same order, whether or not a window is shown;
capturing is a passive copy of what went past.

A block counts as an error when it contains the ``Traceback (most recent call
last)`` header. A traceback arrives in many small ``write()`` calls, so the
collected text is only shown once the output has been quiet for
:data:`QUIET_MS` — otherwise the window would open on the first line and cut
the traceback in half. Capture can happen on any thread; the debounce timer and
the window live on the GUI thread, reached through a queued signal.
"""

import sys
import threading
import traceback

import shiboken6
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QMainWindow,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from syncmoss.constants import AUTHOR_EMAIL, ISSUES_URL

# The header Python writes in front of every traceback, on every platform and
# in every locale — the one reliable "this is an error" marker in a stream that
# also carries ordinary progress prints.
TRACEBACK_MARKER = "Traceback (most recent call last)"

# How long the output has to stay quiet before the collected block is shown.
QUIET_MS = 400

# Upper bound on one collected block. A runaway loop can print tracebacks
# forever; the window must stay openable, so the rest is dropped with a note.
MAX_CHARS = 200000

_lock = threading.Lock()
_pending = []
_pending_chars = 0
_armed = False
_truncated = False

_installed = False
_enabled = True
_suspended = False
_reporter = None
_dialog = None


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
class TeeStream:
    """A write-through copy of a text stream.

    Every call goes to the wrapped stream first (so the terminal is unchanged)
    and only then to :func:`_capture`. ``stream`` may be ``None`` — a GUI build
    started without a console has no ``sys.stdout``/``sys.stderr`` at all, and
    capturing must keep working there, since that is precisely the case where
    the user cannot read the error anywhere else.
    """

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        written = None
        if self.stream is not None:
            try:
                written = self.stream.write(text)
            except Exception:
                written = None
        try:
            _capture(text)
        except Exception:
            pass  # a reporting failure must never break a print()
        return len(text) if written is None else written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        if self.stream is not None:
            try:
                self.stream.flush()
            except Exception:
                pass

    def close(self):
        self.flush()  # never close the real stdio of the process

    def isatty(self):
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def __getattr__(self, name):
        # encoding, errors, fileno, buffer, ... — whatever the wrapped stream has.
        if self.stream is None:
            raise AttributeError(name)
        return getattr(self.stream, name)


def _capture(text):
    """Collect *text* while it is (part of) a traceback. Any thread."""
    global _armed, _pending_chars, _truncated

    if not text or not _enabled or _suspended:
        return

    with _lock:
        if not _armed:
            if TRACEBACK_MARKER not in text:
                return
            _armed = True
        if _pending_chars < MAX_CHARS:
            _pending.append(text)
            _pending_chars += len(text)
        else:
            _truncated = True

    if _reporter is not None:
        _reporter.arrived.emit()


def drain():
    """Take the collected error text out of the buffer and re-arm the capture."""
    global _armed, _pending_chars, _truncated

    with _lock:
        text = ''.join(_pending)
        truncated = _truncated
        _pending.clear()
        _pending_chars = 0
        _armed = False
        _truncated = False

    text = text.strip()
    if truncated and text:
        text += "\n\n[... further output omitted ...]"
    return text


class _Reporter(QObject):
    """Owns the debounce timer and lives on the GUI thread.

    ``arrived`` is emitted from whichever thread wrote to the stream; the queued
    connection is what moves the work to the GUI thread, where a QTimer may be
    started and a window may be created.
    """

    arrived = Signal()

    def __init__(self):
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(QUIET_MS)
        self._timer.timeout.connect(self._flush)
        self.arrived.connect(self._restart, Qt.ConnectionType.QueuedConnection)

    def _restart(self):
        self._timer.start()

    def _flush(self):
        try:
            show_error_report(drain())
        except Exception:
            # A failure here would be printed as a traceback, captured, and
            # bring us straight back — so the reporter steps aside instead.
            set_enabled(False)
            traceback.print_exc(file=sys.__stderr__)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def show_error_report(text, blocking=False):
    """Show *text* in the bug-report window (no-op for empty text).

    A second error while the window is open is appended to it instead of
    stacking another window. The window is modeless by default so the
    application stays usable; ``blocking=True`` is for the one case with no
    event loop to return to (a crash during start-up).
    """
    global _dialog, _suspended

    if not text or not _enabled:
        return None

    app = QApplication.instance()
    if app is None:
        return None  # no GUI (tests, --test, head-less import): terminal only

    # isValid: the C++ window can be gone while the wrapper is still here, and
    # touching it then raises — which _flush would answer by switching the
    # reporter off for the rest of the run.
    if _dialog is not None and shiboken6.isValid(_dialog) and _dialog.isVisible():
        _dialog.append_error(text)
        return _dialog

    # Anything printed while the window is being built is ours, not the user's.
    _suspended = True
    try:
        _dialog = BugReportDialog(text, parent=_report_parent(app))
    finally:
        _suspended = False

    if blocking:
        _dialog.exec()
        dialog, _dialog = _dialog, None
        return dialog

    _dialog.show()
    _dialog.raise_()
    _dialog.activateWindow()
    return _dialog


def _report_parent(app):
    """A main window to own the report, or None — never a passing window.

    ``app.activeWindow()`` itself will not do: it is often a transient one. An
    error printed just before a question box opens (the model's "overwrite?"
    right after a result save, say) reaches the window while that box is
    active, and a report parented to the box is destroyed together with it the
    moment the question is answered. The box's top-level owner is what stays.
    """
    window = app.activeWindow()
    while window is not None and window.parentWidget() is not None:
        window = window.parentWidget().window()
    if isinstance(window, QMainWindow):
        return window
    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow) and widget.isVisible():
            return widget
    return None


def report_exception(blocking=True):
    """Report the exception being handled right now.

    Used where the traceback would otherwise never reach the window: a failure
    before ``QApplication.exec()`` starts, where the debounce timer can never
    fire. The traceback still goes to the terminal first.
    """
    text = traceback.format_exc()
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass
    return show_error_report(drain() or text.strip(), blocking=blocking)


def set_enabled(enabled):
    """Turn reporting on/off ("do not show again" ticks this off for the run)."""
    global _enabled
    _enabled = bool(enabled)
    if not _enabled:
        drain()


def is_enabled():
    return _enabled


def install():
    """Start capturing. Call once, from the GUI thread, after QApplication.

    Returns True when it installed, False when it was already installed.
    """
    global _installed, _reporter

    if _installed:
        return False

    _reporter = _Reporter()
    sys.stdout = TeeStream(sys.stdout)
    sys.stderr = TeeStream(sys.stderr)
    _installed = True
    return True


def uninstall():
    """Undo :func:`install` (used by the tests; harmless if not installed)."""
    global _installed, _reporter

    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name, None)
        if isinstance(stream, TeeStream):
            setattr(sys, name, stream.stream)
    if _reporter is not None:
        _reporter.deleteLater()
        _reporter = None
    drain()
    _installed = False


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
def app_version():
    """The running ``__VERSION__``, or '' when it cannot be determined.

    Imported lazily: ``syncmoss.main`` imports this module, so a top-level
    import would close the circle. Shared with ``supp_menu``.
    """
    try:
        from syncmoss.main import __VERSION__
        return __VERSION__
    except Exception:
        return ""


class BugReportDialog(QDialog):
    """"Report a bug": what happened, what to send, and where to send it."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SYNCmoss - report a bug")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(900, 560)

        layout = QVBoxLayout(self)

        # The address is plain text, not a mailto: link: clicking one would hand
        # the message to whatever mail client is registered on the machine, which
        # is not what people here write from. Copy it, or use the issue tracker.
        version = app_version()
        which_version = f" This is SYNCmoss {version}." if version else ""
        intro = QLabel(
            "<b>Something went wrong.</b> SYNCmoss may well keep working, but "
            "this is a bug and it will not be fixed unless it is reported."
            "<br><br>Please send the error message below to "
            f"<b>{AUTHOR_EMAIL}</b> "
            f'(or open an issue at <a href="{ISSUES_URL}">GitHub</a>), together with:'
            "<ul>"
            "<li>a description of what you were doing — which button or action "
            "caused it, or what you were trying to get;</li>"
            "<li>the spectrum file and the model (.mdl) you were working with;</li>"
            "<li>anything else needed to reproduce it.</li>"
            "</ul>"
            f"{which_version}", self)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setOpenExternalLinks(True)
        layout.addWidget(intro)

        self.error_box = QPlainTextEdit(self)
        self.error_box.setReadOnly(True)
        self.error_box.setPlainText(text)
        self.error_box.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.error_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.error_box, 1)

        self.hint = QLabel("", self)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.silence_box = QCheckBox(
            "Do not show this window again until SYNCmoss is restarted", self)
        layout.addWidget(self.silence_box)

        buttons = QHBoxLayout()
        self.copy_btn = QPushButton("Copy the error message", self)
        self.copy_btn.clicked.connect(self.copy_error)
        self.copy_email_btn = QPushButton("Copy the e-mail address", self)
        self.copy_email_btn.clicked.connect(self.copy_email)
        close_btn = QPushButton("Close", self)
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(self.copy_btn)
        buttons.addWidget(self.copy_email_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def error_text(self):
        return self.error_box.toPlainText()

    def append_error(self, text):
        """Add a later error to the open window instead of opening a second one."""
        if not text:
            return
        self.error_box.appendPlainText("\n" + "-" * 60 + "\n" + text)
        self.error_box.verticalScrollBar().setValue(
            self.error_box.verticalScrollBar().maximum())

    def copy_error(self):
        clipboard = QApplication.clipboard()
        if clipboard is None:
            self.hint.setText("No clipboard available - please select the text and copy it.")
            return False
        clipboard.setText(self.error_text())
        self.hint.setText("Error message copied to the clipboard.")
        return True

    def copy_email(self):
        clipboard = QApplication.clipboard()
        if clipboard is None:
            self.hint.setText("No clipboard available - please select the address and copy it.")
            return False
        clipboard.setText(AUTHOR_EMAIL)
        self.hint.setText(f"{AUTHOR_EMAIL} copied to the clipboard.")
        return True

    def done(self, result):
        # The tick applies to the whole run, not just this window.
        if self.silence_box.isChecked():
            set_enabled(False)
        super().done(result)
