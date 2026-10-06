"""Change spectrum(a) -> Create spectrum from model.

The model of the table, without noise, on the velocities Show model uses -- the
first spectrum's (its own column for a .dat, the calibration for raw counts) or
the Model_<range> grid -- calculated in a thread and saved as a .dat whose
header is the instrumental function it is calculated with: the spectrum's own
.dat header or the one in memory, as the settings say. Pinned here:

* the velocities and the header for every kind of first spectrum;
* the refusals: no spectrum, a missing first file, Nbaseline rows;
* the counts are exactly Show model's model curve, and the whole action --
  thread, save dialog, file -- leaves the path box and the table alone.

ThreadPool stands in for the process pool, so this runs in seconds.
"""
import os
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

import syncmoss.spectrum_io as sio
import syncmoss.syncmoss_main as sm
from syncmoss.instrumental_io import (
    is_data_line, read_dat_metadata_lines, build_dat_metadata_lines,
    resolve_instrumental_for_file, same_method_params,
)
from syncmoss.models import FitInterrupted
from syncmoss.spectrum_io import (
    MODEL_ONLY_POINTS, load_spectrum, model_only_axis, prepare_model_spectrum,
    compute_model_spectrum,
)

pytestmark = [pytest.mark.gui]

ALT_SMS = ["#@INSexp 0.2 0.1 0.7 0.09 -0.06 0.43 -0.54 0.013 0.57", "#@INSint 2.0 0.15"]


def _dat(app, name, header_lines):
    """The data of the tests' calibration spectrum under *header_lines*."""
    with open(app.calibration_path) as src:
        data = [line for line in src if line.strip() and is_data_line(line.strip())]
    path = os.path.join(os.path.dirname(app.calibration_path), name)
    with open(path, "w") as out:
        out.writelines(line + "\n" for line in header_lines)
        out.writelines(data)
    return path


def _model_of_one_sextet(app, box):
    app.jn0_input.setText("16")        # keeps the transmission integral cheap
    app.process_path.setPlainText(box)
    app.params_table.select_model(1, "Sextet")


def _pump_until(condition, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QCoreApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture
def pool(physics_app):
    physics_app.pool = ThreadPool(2)
    yield physics_app.pool
    physics_app.pool.terminate()
    physics_app.pool.join()


# ---------------------------------------------------------------------------
# What is calculated: velocities and header
# ---------------------------------------------------------------------------

def test_a_dat_gives_its_velocities_and_its_own_header(physics_app):
    dat = _dat(physics_app, "own.dat", ["# Converted by SYNCMoss"] + ALT_SMS)
    _model_of_one_sextet(physics_app, repr([dat]))
    job = prepare_model_spectrum(physics_app)
    assert job is not None, physics_app.log.toPlainText()
    np.testing.assert_array_equal(job['A'], load_spectrum(physics_app, [dat])[0][0])
    assert job['method_params']['source'] == 'file'
    assert job['header'][:-1] == ALT_SMS
    assert job['header'][-1].startswith("# Model spectrum without noise")
    assert job['default_name'] == "own_model.dat"


@pytest.mark.parametrize("why", ["no header", "header not used"])
def test_otherwise_the_header_is_the_one_in_memory(physics_app, why):
    dat = _dat(physics_app, "memory.dat", [] if why == "no header" else ALT_SMS)
    physics_app.use_dat_instrumental_metadata = why == "no header"
    _model_of_one_sextet(physics_app, repr([dat]))
    job = prepare_model_spectrum(physics_app)
    assert job['method_params']['source'] == 'internal'
    assert job['header'][:-1] == build_dat_metadata_lines(physics_app)[0]


def test_the_first_of_several_spectra_is_used(physics_app, tmp_path):
    """Only the first one has to exist."""
    first = _dat(physics_app, "first.dat", ALT_SMS)
    _model_of_one_sextet(physics_app, repr([first, str(tmp_path / "not_there.dat")]))
    job = prepare_model_spectrum(physics_app)
    assert job is not None, physics_app.log.toPlainText()
    assert job['source'] == "first.dat"
    assert job['header'][:-1] == ALT_SMS


def test_raw_counts_take_the_velocities_of_the_calibration(physics_app, tmp_path):
    """An .mca holds counts only: the velocities are the calibration's."""
    calibration_axis = sio._read_calibration_axis(physics_app.calibration_path, False)[0]
    raw = tmp_path / "raw.mca"
    raw.write_text("@A " + " ".join(["1000"] * (2 * len(calibration_axis))) + "\n")
    _model_of_one_sextet(physics_app, repr([str(raw)]))
    job = prepare_model_spectrum(physics_app)
    assert job is not None, physics_app.log.toPlainText()
    np.testing.assert_array_equal(job['A'], calibration_axis)
    assert job['method_params']['source'] == 'internal'
    assert job['header'][:-1] == build_dat_metadata_lines(physics_app)[0]


def test_model_only_takes_the_show_model_grid(physics_app):
    _model_of_one_sextet(physics_app, "Model_6")
    job = prepare_model_spectrum(physics_app)
    assert job is not None, physics_app.log.toPlainText()
    assert len(job['A']) == MODEL_ONLY_POINTS
    np.testing.assert_array_equal(job['A'], model_only_axis(6.0))
    assert job['A'][0] == -6.0 and job['A'][-1] == 6.0
    assert job['header'][:-1] == build_dat_metadata_lines(physics_app)[0]
    assert job['default_name'] == "spectrum_Model_6.dat"


# ---------------------------------------------------------------------------
# Refusals: said in the log, nothing calculated
# ---------------------------------------------------------------------------

def test_no_spectrum(physics_app):
    _model_of_one_sextet(physics_app, "")
    assert prepare_model_spectrum(physics_app) is None
    assert physics_app.log.toPlainText() == "No spectrum selected"


def test_a_missing_first_spectrum(physics_app, tmp_path):
    _model_of_one_sextet(physics_app, repr([str(tmp_path / "not_there.dat")]))
    assert prepare_model_spectrum(physics_app) is None
    assert "Not a spectrum file" in physics_app.log.toPlainText()


def test_a_model_with_nbaseline(physics_app):
    _model_of_one_sextet(physics_app, repr([physics_app.calibration_path]))
    physics_app.params_table.select_model(2, "Nbaseline")
    physics_app.params_table.select_model(3, "Sextet")
    assert prepare_model_spectrum(physics_app) is None
    assert "Nbaseline" in physics_app.log.toPlainText()


def test_refused_while_a_calculation_runs(physics_app, monkeypatch):
    called = []
    monkeypatch.setattr(sm, "prepare_model_spectrum", lambda app: called.append(app))
    physics_app.inprogress = True
    physics_app.busy_with = 'Fitting'
    try:
        physics_app.create_spectrum_from_model()
    finally:
        physics_app.inprogress = False
    assert called == []
    assert "not available right now" in physics_app.log.toPlainText()


# ---------------------------------------------------------------------------
# The calculation and the whole action
# ---------------------------------------------------------------------------

def test_the_counts_are_the_show_model_curve(physics_app, pool):
    dat = _dat(physics_app, "curve.dat", ALT_SMS)
    _model_of_one_sextet(physics_app, repr([dat]))
    assert physics_app.initialize_parameters()
    shown = []
    thread = sm.ShowModelThread(physics_app, [dat], pool)
    thread.finished.connect(lambda *args: shown.append(args))
    thread.error.connect(lambda message: shown.append(message))
    thread.run()        # synchronous
    assert shown and not isinstance(shown[0], str), shown
    A, SPC_f = shown[0][0], shown[0][2]

    job = prepare_model_spectrum(physics_app)
    counts = compute_model_spectrum(job, pool)
    np.testing.assert_array_equal(job['A'], A)
    np.testing.assert_array_equal(counts, SPC_f)


def test_create_spectrum_from_model_end_to_end(physics_app, pool, tmp_path, monkeypatch):
    dat = _dat(physics_app, "measured.dat", ["# Converted by SYNCMoss"] + ALT_SMS)
    box = repr([dat])
    _model_of_one_sextet(physics_app, box)
    out = str(tmp_path / "created.dat")
    offered = []
    monkeypatch.setattr(sio.QFileDialog, "getSaveFileName", staticmethod(
        lambda parent, caption, directory, file_filter: offered.append(
            (directory, file_filter)) or (out, "")))
    table_before = sm.read_model(physics_app)[1].copy()

    physics_app.create_spectrum_from_model()
    assert physics_app.inprogress, physics_app.log.toPlainText()
    _pump_until(lambda: not physics_app.inprogress)
    physics_app.model_spectrum_thread.wait()

    log = physics_app.log.toPlainText()
    assert os.path.exists(out), log
    assert "saved to created.dat" in log and "no noise" in log
    assert offered[0][0].endswith("measured_model.dat") and offered[0][1] == "DAT files (*.dat)"
    # The instrumental function the model was calculated with travels in the
    # file, so a fit of it uses the same one
    assert read_dat_metadata_lines(out) == ALT_SMS
    assert same_method_params(resolve_instrumental_for_file(physics_app, out),
                              resolve_instrumental_for_file(physics_app, dat))
    # Velocities of the spectrum, counts of the model (the file holds 6 decimals)
    A_list, B_list = load_spectrum(physics_app, [out])
    job = prepare_model_spectrum(physics_app)
    np.testing.assert_allclose(A_list[0], job['A'], rtol=0, atol=1e-6)
    np.testing.assert_allclose(B_list[0], compute_model_spectrum(job, pool), rtol=0, atol=1e-6)
    assert np.ptp(B_list[0]) > 0
    # Not loaded: the path box and the table are as they were
    assert physics_app.process_path.toPlainText() == box
    np.testing.assert_array_equal(sm.read_model(physics_app)[1], table_before)


def test_canceling_the_save_dialog_writes_nothing(physics_app, pool, monkeypatch):
    _model_of_one_sextet(physics_app, "Model_3")
    written = []
    monkeypatch.setattr(sio.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(sio, "save_spectrum_with_metadata", lambda *a: written.append(a))
    physics_app.create_spectrum_from_model()
    _pump_until(lambda: not physics_app.inprogress)
    physics_app.model_spectrum_thread.wait()
    assert physics_app.log.toPlainText() == "Save canceled"
    assert written == []


def test_the_thread_reports_the_interrupt_quietly(monkeypatch, capsys):
    def interrupted(job, pool):
        raise FitInterrupted()
    monkeypatch.setattr(sm, "compute_model_spectrum", interrupted)
    got = []
    thread = sm.ModelSpectrumThread({}, None)
    thread.error.connect(lambda *a: got.append(a))
    thread.start()
    thread.wait(5000)
    QCoreApplication.processEvents()
    assert got == [("Interrupted by the user",)]
    assert 'Traceback' not in capsys.readouterr().out
