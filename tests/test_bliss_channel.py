"""Reading the spectrum online from a Bliss channel (optional, Linux only).

A ``McaAcq_channel_...`` name in the path box is read from Bliss and folded like
the counts of an ``.mca`` file. ``bliss`` is not a dependency and exists for
Linux only, so here ``Channel`` is replaced by a fake serving fixed counts.
Pinned:

  * a channel is folded exactly like the same counts in an .mca file;
  * it is read on the main thread only -- by the path check, which runs there
    before any worker thread starts -- and afresh every time; the worker threads
    get the counts read there and never touch Bliss;
  * a channel that cannot be read is refused, by name;
  * if bliss cannot be imported nothing happens at all: the name is then an
    ordinary missing path, with the ordinary message.
"""
import importlib
import os
import sys
import threading
import types

import numpy as np
import pytest

from syncmoss import bliss_channel
from syncmoss.Calibration import _load_raw_counts
from syncmoss.spectrum_io import load_spectrum

_MCA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alpha_fe_sms_000.mca")
NAME = "McaAcq_channel_test"


class FakeChannel:
    """Stands in for ``bliss.config.channels.Channel``: a name reads as its entry
    in ``values`` (None for any other name), and every read is logged together
    with the thread it came from."""
    values = {}
    reads = []

    def __init__(self, name):
        self.name = name

    @property
    def value(self):
        FakeChannel.reads.append((self.name, threading.current_thread()))
        return FakeChannel.values.get(self.name)


@pytest.fixture
def bliss(monkeypatch):
    """Bliss available, its channel NAME holding the raw counts of the SMS
    fixture spectrum."""
    monkeypatch.setattr(FakeChannel, "values", {NAME: _load_raw_counts(_MCA).tolist()})
    monkeypatch.setattr(FakeChannel, "reads", [])
    monkeypatch.setattr(bliss_channel, "Channel", FakeChannel)
    monkeypatch.setattr(bliss_channel, "AVAILABLE", True)
    monkeypatch.setattr(bliss_channel, "_last_counts", {})
    return FakeChannel


def _calibration(tmp_path):
    """A Calibration.dat for the 1024-channel fixture: 'sin' folding of
    channels 1..1023 gives 511 velocity points."""
    path = tmp_path / "Calibration.dat"
    rows = "".join(f"{v:.6f}\t1\n" for v in np.linspace(-7, 7, 511))
    path.write_text("#\tsin \t1\t1023\n" + rows)
    return str(path)


def _in_worker(function):
    """What ``function()`` returns (or raises) on a worker thread, where the
    show-model and fitting threads run."""
    result = []

    def run():
        try:
            result.append(function())
        except Exception as e:
            result.append(e)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    return result[0]


def _path_check(paths):
    """The check Show spectrum, Show model and Fit run on the main thread
    before any worker thread starts."""
    from syncmoss.syncmoss_main import PhysicsApp
    return PhysicsApp.check_spectrum_paths_exist(paths)


# --- the spectrum -------------------------------------------------------------

def test_a_channel_is_folded_like_the_same_counts_in_an_mca_file(bliss, tmp_path):
    calibration = _calibration(tmp_path)
    bliss_channel.refresh(NAME)
    (A_file,), (B_file,) = load_spectrum(None, [_MCA], calibration_path=calibration)
    (A_chan,), (B_chan,) = load_spectrum(None, [NAME], calibration_path=calibration)
    assert len(B_file) == 511
    np.testing.assert_array_equal(A_chan, A_file)
    np.testing.assert_array_equal(B_chan, B_file)


def test_every_refresh_reads_the_channel_again(bliss):
    """The spectrum keeps accumulating: each Show or Fit must see the new one."""
    bliss_channel.refresh(NAME)
    bliss.values[NAME] = [1.0] * 1024
    assert bliss_channel.refresh(NAME).sum() == 1024
    assert bliss_channel.counts(NAME).sum() == 1024


def test_a_throw_away_channel_is_read_first(bliss):
    """The original program's recipe -- see bliss_channel.refresh."""
    bliss_channel.refresh(NAME)
    names = [name for name, _ in bliss.reads]
    assert len(names) == 2 and names[0] != NAME and names[1] == NAME


# --- threads --------------------------------------------------------------------

def test_worker_threads_get_the_counts_read_on_the_main_thread(bliss):
    read_here = bliss_channel.refresh(NAME)
    reads = len(bliss.reads)
    np.testing.assert_array_equal(_in_worker(lambda: bliss_channel.counts(NAME)), read_here)
    assert len(bliss.reads) == reads, "a worker thread touched Bliss"


def test_a_worker_thread_never_reads_bliss_itself(bliss):
    assert _in_worker(lambda: bliss_channel.counts(NAME)) is None
    assert isinstance(_in_worker(lambda: bliss_channel.refresh(NAME)), RuntimeError)
    assert bliss.reads == []


def test_the_show_model_thread_keeps_a_channel_name_as_it_is(bliss):
    """It makes every path absolute first, which would hide the channel name."""
    assert bliss_channel.abspath(NAME) == NAME
    assert bliss_channel.abspath("a.dat") == os.path.abspath("a.dat")


# --- the path check -----------------------------------------------------------

def test_the_path_check_reads_the_channel(bliss):
    assert _path_check([NAME]) is None
    assert [name for name, _ in bliss.reads][-1] == NAME
    assert bliss_channel.counts(NAME) is not None


def test_a_channel_without_a_value_is_refused_by_name(bliss):
    message = _path_check(["McaAcq_channel_typo"])
    assert message.startswith("Could not read Bliss channel McaAcq_channel_typo")


def test_channels_and_files_are_checked_together(bliss, tmp_path):
    """Nbaseline fits can mix the online spectrum with files."""
    real = tmp_path / "a.dat"
    real.write_text("-1 1\n1 1\n")
    assert _path_check([str(real), NAME]) is None
    assert "missing.dat" in _path_check([NAME, str(tmp_path / "missing.dat")])


def test_without_bliss_a_channel_name_is_just_a_missing_file(monkeypatch):
    monkeypatch.setattr(bliss_channel, "AVAILABLE", False)
    assert not bliss_channel.is_channel(NAME)
    assert _path_check([NAME]) == _path_check(["no_such_file"]).replace("no_such_file", NAME)


# --- importing bliss -----------------------------------------------------------

@pytest.fixture
def reimport(monkeypatch):
    """Runs bliss_channel's import of bliss again; afterwards the real
    environment is restored and the module re-imported in it."""
    monkeypatch.setenv("BEACON_HOST", "")  # recorded first, so the original
    monkeypatch.delenv("BEACON_HOST")      # state is restored afterwards
    yield lambda: importlib.reload(bliss_channel)
    monkeypatch.undo()
    importlib.reload(bliss_channel)


def test_without_bliss_nothing_happens(monkeypatch, reimport):
    for name in ("bliss", "bliss.config", "bliss.config.channels"):
        monkeypatch.setitem(sys.modules, name, None)  # cannot be imported
    reimport()
    assert bliss_channel.AVAILABLE is False
    assert "BEACON_HOST" not in os.environ


@pytest.mark.parametrize("preset, expected", [
    (None, "id14:25000"), ("id99:25000", "id99:25000")])
def test_with_bliss_the_beacon_host_defaults_to_id14(monkeypatch, reimport, preset, expected):
    for name in ("bliss", "bliss.config", "bliss.config.channels"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["bliss.config.channels"].Channel = FakeChannel
    if preset:
        monkeypatch.setenv("BEACON_HOST", preset)
    reimport()
    assert bliss_channel.AVAILABLE is True and bliss_channel.Channel is FakeChannel
    assert os.environ["BEACON_HOST"] == expected
