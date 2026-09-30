"""The "! INTERRUPT !" button.

Every calculation that uses the pool goes through ``models.TI``, whose pool
calls (``models._pool_starmap``) give up as soon as the flag is set. So the
fit, sequential fit, Show model, calibration and both instrumental searches
stop within ~50 ms without anything being killed; the pool is killed only if
its workers are still busy a second later. Before this, a fit waiting on a
terminated pool stayed blocked for good (``Pool.terminate`` never completes the
pending results).

Checked here: the helper gives the same results as ``pool.starmap``; it stops
on the flag and on a terminated pool; each calculation thread reports the
interrupt quietly (no traceback -- the bug reporter would take it for a crash);
the button ignores extra clicks, touches nothing when idle, kills the pool only
when needed; and the app works normally afterwards.

ThreadPool stands in for the process pool (except in one test that needs a
real terminate), so this runs in seconds.
"""
import multiprocessing as mp
import threading
import time
from multiprocessing.pool import ThreadPool

import pytest
from PySide6.QtCore import QCoreApplication, QThread

import syncmoss.models as m5
import syncmoss.syncmoss_main as sm
from syncmoss.models import FIT_CANCEL, FitInterrupted, _pool_starmap


@pytest.fixture(autouse=True)
def _clear_flag():
    FIT_CANCEL.clear()
    yield
    FIT_CANCEL.clear()


@pytest.fixture
def thread_pool():
    pool = ThreadPool(2)
    yield pool
    pool.terminate()
    pool.join()


class _SerialPool:
    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _mul(x, y):
    return x * y


def _pump_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QCoreApplication.processEvents()
        time.sleep(0.01)


def _in_thread(func):
    """Run func in a daemon thread; return (thread, outcome list)."""
    outcome = []

    def run():
        try:
            outcome.append(func())
        except FitInterrupted:
            outcome.append('interrupted')

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, outcome


# ------------------------------------------------------------ the pool helper

def test_helper_gives_the_same_results_as_starmap(thread_pool):
    args = [(i, i + 0.5) for i in range(20)]
    assert _pool_starmap(thread_pool, _mul, args) == thread_pool.starmap(_mul, args)
    assert _pool_starmap(_SerialPool(), _mul, args) == thread_pool.starmap(_mul, args)


def test_helper_passes_worker_errors_through(thread_pool):
    with pytest.raises(ZeroDivisionError):
        _pool_starmap(thread_pool, lambda x: 1 / x, [(0,)])


def test_helper_refuses_new_work_when_the_flag_is_set(thread_pool):
    FIT_CANCEL.set()
    with pytest.raises(FitInterrupted):
        _pool_starmap(thread_pool, _mul, [(1, 2)])
    with pytest.raises(FitInterrupted):
        _pool_starmap(_SerialPool(), _mul, [(1, 2)])
    assert not thread_pool._cache, "nothing was handed to the workers"


def test_the_flag_releases_a_waiting_thread_at_once(thread_pool):
    t, outcome = _in_thread(lambda: _pool_starmap(thread_pool, time.sleep, [(0.5,)]))
    time.sleep(0.1)
    started = time.monotonic()
    FIT_CANCEL.set()
    t.join(1)
    assert outcome == ['interrupted']
    assert time.monotonic() - started < 0.3


def test_a_terminated_pool_does_not_leave_the_thread_stuck():
    """The original bug, with a real process pool: plain starmap waits for ever."""
    pool = mp.Pool(1)
    try:
        t, outcome = _in_thread(lambda: _pool_starmap(pool, time.sleep, [(5,)]))
        time.sleep(0.3)
        pool.terminate()
        t.join(2)
        assert outcome == ['interrupted'] and not t.is_alive()
    finally:
        pool.terminate()
        pool.join()


# ------------------------------------------ each thread reports it quietly

def _run_thread(thread, signal_name):
    got = []
    getattr(thread, signal_name).connect(lambda *a: got.append(a))
    thread.start()
    thread.wait(5000)
    QCoreApplication.processEvents()
    return got


def _interrupted(*_a, **_k):
    raise FitInterrupted()


def test_fitting_thread_reports_the_interrupt_quietly(physics_app, monkeypatch, capsys):
    monkeypatch.setattr(sm.fitting_io, 'fit_single_spectrum', _interrupted)
    got = _run_thread(sm.FittingThread(physics_app, 'x.dat', None), 'error')
    assert got == [("Interrupted by the user",)]
    assert 'Traceback' not in capsys.readouterr().out


def test_instrumental_thread_reports_the_interrupt_quietly(physics_app, monkeypatch, capsys):
    monkeypatch.setattr(sm, 'instrumental', _interrupted)
    got = _run_thread(sm.InstrumentalThread(physics_app, 0, 0, None), 'error')
    assert got == [("Interrupted by the user",)]
    assert 'Traceback' not in capsys.readouterr().out


def test_calibration_thread_reports_the_interrupt_quietly(monkeypatch, capsys):
    monkeypatch.setattr(sm, 'Calibration', _interrupted)
    got = _run_thread(sm.CalibrationThread('.', 'x.mca', 1, [], 16, 0, 1, 0, None), 'error')
    assert got == [("Interrupted by the user",)]
    assert 'Traceback' not in capsys.readouterr().out


def test_show_model_thread_reports_the_interrupt_quietly(physics_app, monkeypatch, capsys):
    monkeypatch.setattr(sm, 'read_model', _interrupted)
    got = _run_thread(sm.ShowModelThread(physics_app, [], None), 'error')
    assert got == [("Interrupted by the user",)]
    assert 'Traceback' not in capsys.readouterr().out


def test_sequential_fit_stops_at_the_interrupt(physics_app, monkeypatch):
    calls = []

    def fit(app, spectrum_file, pool, **kw):
        calls.append(spectrum_file)
        raise FitInterrupted()

    monkeypatch.setattr(sm.fitting_io, 'fit_single_spectrum', fit)
    got = _run_thread(sm.SequentialFittingThread(
        physics_app, ['a', 'b', 'c'], None, 0, [0, 0, 0]), 'finished')
    assert calls == ['a'], "the rest of the batch is not attempted"
    assert got[0][0]['succeeded'] == 0


def test_fit_single_spectrum_does_not_swallow_the_interrupt(physics_app, monkeypatch):
    """It turns every other error into success=False (+ traceback); the
    interrupt must reach the thread instead."""
    monkeypatch.setattr(sm.fitting_io, 'read_model_full', _interrupted)
    with pytest.raises(FitInterrupted):
        sm.fitting_io.fit_single_spectrum(physics_app, physics_app.calibration_path, None)


# ---------------------------------------------------------------- the button

class _PoolThread(QThread):
    """Stands in for a calculation: one pool call, like TI makes."""

    def __init__(self, pool, seconds):
        super().__init__()
        self.pool = pool
        self.seconds = seconds
        self.outcome = None

    def run(self):
        try:
            _pool_starmap(self.pool, time.sleep, [(self.seconds,)])
            self.outcome = 'returned'
        except FitInterrupted:
            self.outcome = 'interrupted'


def _start(app, pool, seconds):
    app.pool = pool
    app.inprogress = True
    app.busy_with = 'Fitting'
    app.fitting_thread = _PoolThread(pool, seconds)
    app.fitting_thread.start()
    _pump_until(lambda: bool(pool._cache))
    return app.fitting_thread


def test_interrupt_with_nothing_running_touches_nothing(physics_app):
    physics_app.interrupt()
    assert physics_app.pool is None, "no pool created or killed"
    assert not FIT_CANCEL.is_set()
    assert not physics_app._interrupting


def test_gentle_interrupt_multi_click_and_further_work(physics_app, thread_pool):
    thread = _start(physics_app, thread_pool, 0.3)
    for _ in range(6):                               # the multi-click
        physics_app.interrupt()
    assert physics_app._interrupting
    assert physics_app._reject_if_busy('Fitting'), "no new work while stopping"

    _pump_until(lambda: thread.outcome is not None)
    assert thread.outcome == 'interrupted'
    _pump_until(lambda: not physics_app._interrupting)
    assert physics_app.pool is thread_pool, "workers finished in time: not killed"
    assert "Interrupted" in physics_app.log.toPlainText()
    assert not FIT_CANCEL.is_set() and not physics_app.inprogress
    assert not physics_app._reject_if_busy('Fitting')
    # and the app works normally afterwards
    assert _pool_starmap(physics_app.pool, _mul, [(3, 4)]) == [12]


def test_workers_still_busy_after_a_second_are_replaced(physics_app, thread_pool, monkeypatch):
    new_pools = []
    monkeypatch.setattr(sm.mp, 'Pool', lambda processes: new_pools.append(ThreadPool(1)) or new_pools[-1])
    try:
        _start(physics_app, thread_pool, 2.0)
        physics_app.interrupt()
        _pump_until(lambda: not physics_app._interrupting)
        assert physics_app.pool is new_pools[0]
        assert "pool terminated and recreated" in physics_app.log.toPlainText()
        assert not FIT_CANCEL.is_set() and not physics_app.inprogress
        assert _pool_starmap(physics_app.pool, _mul, [(2, 5)]) == [10]
    finally:
        for p in new_pools:
            p.terminate()
            p.join()


def test_a_latched_busy_flag_is_released(physics_app):
    """An entry point that set inprogress and returned early: Interrupt is the
    way out, and must leave nothing set behind."""
    physics_app.inprogress = True
    physics_app.interrupt()
    _pump_until(lambda: not physics_app._interrupting)
    assert not physics_app.inprogress and not FIT_CANCEL.is_set()


# ------------------------------------------------ a real fit, end to end

def _prepare_sextet_fit(app):
    app.jn0_input.setText("16")
    spectrum = app.calibration_path
    app.process_path.setPlainText(repr([spectrum]))
    app.path_list = [spectrum]
    app.show_pressed()
    app.params_table.select_model(1, "Sextet")
    assert app.initialize_parameters()
    return spectrum


def test_a_real_fit_is_interrupted_and_the_next_one_works(physics_app, thread_pool):
    spectrum = _prepare_sextet_fit(physics_app)
    FIT_CANCEL.set()
    with pytest.raises(FitInterrupted):
        sm.fitting_io.fit_single_spectrum(physics_app, spectrum, thread_pool)
    FIT_CANCEL.clear()
    result = sm.fitting_io.fit_single_spectrum(physics_app, spectrum, thread_pool)
    assert result['success'], result.get('message')


def test_the_button_during_a_real_fit_in_the_app(physics_app, thread_pool, capsys):
    """Fit, Interrupt (twice), then Fit again: the second fit completes."""
    _prepare_sextet_fit(physics_app)
    physics_app.pool = thread_pool
    physics_app.fit_pressed()
    assert physics_app.inprogress and physics_app.fitting_thread.isRunning()
    physics_app.interrupt()
    physics_app.interrupt()
    _pump_until(lambda: not physics_app._interrupting, timeout=30)
    assert not physics_app.fitting_thread.isRunning()
    assert physics_app.log.toPlainText().startswith("Interrupted")
    assert 'Traceback' not in capsys.readouterr().out

    physics_app.results_table.current_chi2 = None
    physics_app.fit_pressed()
    _pump_until(lambda: not physics_app.inprogress, timeout=60)
    assert physics_app.results_table.current_chi2 is not None, physics_app.log.toPlainText()
