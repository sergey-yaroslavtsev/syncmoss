"""console_io: what gives the windowed Windows build (the only executable of
the bundle) a working stdout, and how main.py takes its switches out of argv.
Ported from the SYNCtime branch's tests/test_launcher.py.

The terminal cases (AttachConsole, AllocConsole) need a real windowed process
started from a terminal; the frozen --test in CI starts it redirected.
"""
import os
import sys

import pytest

from syncmoss import console_io


def test_consume_flag_removes_every_occurrence():
    argv = ['app', '--console', 'x', '--console']
    assert console_io.consume_flag(argv, '--console') is True
    assert argv == ['app', 'x']
    assert console_io.consume_flag(argv, '--console') is False


def test_consume_flag_accepts_several_names():
    argv = ['app', '-V']
    assert console_io.consume_flag(argv, '--version', '-V') is True
    assert argv == ['app']


def test_consume_flag_is_case_insensitive_and_exact():
    argv = ['app', '--Test', '--test-again', 'x']
    assert console_io.consume_flag(argv, '--test') is True
    assert argv == ['app', '--test-again', 'x']          # not a prefix match


def test_enable_console_leaves_working_streams_alone():
    """The case that must never break: a normal interpreter (or a redirected
    frozen build) already has stdio, so enable_console must not replace it."""
    before = (sys.stdout, sys.stderr)
    assert console_io.enable_console() is True
    assert (sys.stdout, sys.stderr) == before


@pytest.mark.skipif(sys.platform != 'win32', reason="elsewhere a GUI binary keeps the stdio it inherited")
def test_a_windowed_build_with_redirected_output_gets_streams(monkeypatch):
    """The frozen --test in CI: started redirected, descriptor 1 is valid but
    a windowed build may have no sys.stdout on top of it."""
    assert os.fstat(1)                                     # valid here (pytest's capture)
    for name in ('stdin', 'stdout', 'stderr'):
        monkeypatch.setattr(sys, name, getattr(sys, name))  # restored afterwards
    monkeypatch.setattr(sys, 'stdout', None)
    monkeypatch.setattr(sys, 'stderr', None)
    assert console_io.enable_console() is True
    assert sys.stdout is not None and sys.stderr is not None
    assert sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2
    print("Mössbauer δ", file=sys.stdout)                 # writes, in UTF-8
    sys.stdout.flush()
