"""The global error capture and the "report a bug" window.

The capture has to satisfy two requirements that pull against each other: it
must not change what the terminal shows (write-through, always, first), and it
must still notice an error printed by code that caught the exception itself.
The tests below pin both, plus the traceback-only trigger — the stream also
carries ordinary progress prints and must not raise a window for those.
"""
import sys

import pytest
import shiboken6

from syncmoss import error_reporter as er
from syncmoss.constants import AUTHOR_EMAIL


TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "syncmoss_main.py", line 1, in save_result_pressed\n'
    "    boom()\n"
    "ValueError: boom\n"
)


class FakeStream:
    """Minimal text stream that records what was written through it."""

    def __init__(self):
        self.chunks = []
        self.flushed = 0
        self.encoding = 'utf-8'

    def write(self, text):
        self.chunks.append(text)
        return len(text)

    def flush(self):
        self.flushed += 1

    def isatty(self):
        return False

    def text(self):
        return ''.join(self.chunks)


@pytest.fixture(autouse=True)
def clean_reporter_state():
    """Every test starts with an empty buffer and reporting enabled."""
    er.drain()
    er.set_enabled(True)
    yield
    er.drain()
    er.set_enabled(True)


def test_tee_writes_everything_through_and_captures_nothing_ordinary():
    stream = FakeStream()
    tee = er.TeeStream(stream)

    tee.write("Fitting started\n")
    tee.write("chi2 = 1.03\n")

    assert stream.text() == "Fitting started\nchi2 = 1.03\n"
    assert er.drain() == ""


def test_tee_captures_a_traceback_and_still_prints_it():
    stream = FakeStream()
    tee = er.TeeStream(stream)

    tee.write(TRACEBACK)

    assert stream.text() == TRACEBACK          # terminal output is unchanged
    assert "ValueError: boom" in er.drain()


def test_a_traceback_split_over_many_writes_is_collected_whole():
    """traceback.print_exc() emits one write per line — none may be lost."""
    tee = er.TeeStream(FakeStream())
    for line in TRACEBACK.splitlines(keepends=True):
        tee.write(line)

    captured = er.drain()
    assert captured.startswith("Traceback (most recent call last):")
    assert "save_result_pressed" in captured
    assert captured.endswith("ValueError: boom")


def test_a_caught_and_printed_traceback_is_captured():
    """The app's own `print(f"...: {e}\\n{traceback.format_exc()}")` pattern."""
    tee = er.TeeStream(FakeStream())
    tee.write("Error saving results: boom\n" + TRACEBACK)

    captured = er.drain()
    assert captured.startswith("Error saving results: boom")
    assert "ValueError: boom" in captured


def test_drain_re_arms_the_capture():
    tee = er.TeeStream(FakeStream())
    tee.write(TRACEBACK)
    assert er.drain()
    tee.write("ordinary output\n")
    assert er.drain() == ""            # not still collecting after the first block


def test_capture_is_bounded():
    tee = er.TeeStream(FakeStream())
    tee.write(TRACEBACK)
    tee.write("x" * (er.MAX_CHARS + 10))
    tee.write("y" * 100)

    captured = er.drain()
    assert len(captured) < er.MAX_CHARS + 5000
    assert captured.endswith("[... further output omitted ...]")
    assert "y" * 100 not in captured


def test_disabled_reporter_captures_nothing_but_still_prints():
    stream = FakeStream()
    tee = er.TeeStream(stream)
    er.set_enabled(False)

    tee.write(TRACEBACK)

    assert stream.text() == TRACEBACK
    assert er.drain() == ""


def test_tee_survives_a_missing_stream():
    """A frozen GUI build started without a console has sys.stderr is None."""
    tee = er.TeeStream(None)
    assert tee.write(TRACEBACK) == len(TRACEBACK)
    tee.flush()
    assert tee.isatty() is False
    assert "ValueError: boom" in er.drain()


@pytest.mark.gui
def test_install_and_uninstall_restore_the_original_streams(qapp):
    original_out, original_err = sys.stdout, sys.stderr
    try:
        assert er.install() is True
        assert er.install() is False      # idempotent
        assert isinstance(sys.stdout, er.TeeStream)
        assert isinstance(sys.stderr, er.TeeStream)
        assert sys.stdout.stream is original_out
    finally:
        er.uninstall()
    assert sys.stdout is original_out
    assert sys.stderr is original_err


def test_the_window_never_launches_a_mail_client():
    """No mailto: here either — the address is shown and copied, not handed off."""
    assert not hasattr(er, 'QDesktopServices')


@pytest.mark.gui
def test_bug_report_dialog_shows_the_error_and_copies_it(qapp):
    dialog = er.BugReportDialog(TRACEBACK)
    try:
        assert "ValueError: boom" in dialog.error_text()

        assert dialog.copy_error() is True
        assert qapp.clipboard().text() == dialog.error_text()
        assert "clipboard" in dialog.hint.text().lower()

        assert dialog.copy_email() is True
        assert qapp.clipboard().text() == AUTHOR_EMAIL

        dialog.append_error("Traceback: a second one\nKeyError: k")
        assert "ValueError: boom" in dialog.error_text()
        assert "KeyError: k" in dialog.error_text()
    finally:
        dialog.deleteLater()


@pytest.mark.gui
def test_do_not_show_again_switches_reporting_off_for_the_run(qapp):
    dialog = er.BugReportDialog(TRACEBACK)
    try:
        dialog.silence_box.setChecked(True)
        dialog.reject()
        assert er.is_enabled() is False
    finally:
        er.set_enabled(True)
        dialog.deleteLater()


@pytest.mark.gui
def test_show_error_report_appends_to_an_open_window(qapp):
    first = er.show_error_report(TRACEBACK)
    try:
        assert first is not None
        second = er.show_error_report("Traceback ...\nKeyError: k")
        assert second is first          # one window, not a stack of them
        assert "KeyError: k" in first.error_text()
    finally:
        if first is not None:
            first.close()
            first.deleteLater()
        er._dialog = None


@pytest.mark.gui
def test_show_error_report_ignores_empty_text(qapp):
    assert er.show_error_report("") is None


@pytest.mark.gui
def test_the_report_belongs_to_the_main_window_not_a_passing_one(qapp):
    """An error printed just before a question box opens reaches the window
    while that box is active. Parented to it, the report died the moment the
    question was answered — and the next error then met a dead wrapper."""
    from PySide6.QtWidgets import QMainWindow, QMessageBox

    main = QMainWindow()
    main.show()
    box = QMessageBox(main)
    box.show()
    box.activateWindow()
    for _ in range(5):
        qapp.processEvents()
    try:
        assert qapp.activeWindow() is box          # the situation under test
        report = er.show_error_report(TRACEBACK)
        assert report.parent() is main
        shiboken6.delete(box)                     # the question is answered
        assert shiboken6.isValid(report)
        assert er.show_error_report("Traceback ...\nKeyError: k") is report
        assert er.is_enabled()
    finally:
        er._dialog = None
        shiboken6.delete(main)


@pytest.mark.gui
def test_a_destroyed_report_window_does_not_switch_reporting_off(qapp):
    stale = er.BugReportDialog(TRACEBACK)
    stale.show()
    shiboken6.delete(stale)
    er._dialog = stale
    fresh = None
    try:
        fresh = er.show_error_report(TRACEBACK)
        assert fresh is not None and fresh is not stale
        assert er.is_enabled()
    finally:
        er._dialog = None
        if fresh is not None:
            shiboken6.delete(fresh)
