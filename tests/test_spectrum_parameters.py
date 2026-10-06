"""Parameters of the spectra: N, N1, N2, ... in the formulas of a sequence fit.

N is the spectrum's number in the sequence (its position in the path box,
from 1); N1, N2, ... are numbers given with it, in the path box as
('file', N1, N2, ...) or loaded from a parameters file (#basename / #number
are reserved names there). The names are replaced by their values in the
Expression/Distr/Corr texts before anything evaluates them.
"""
import os
import shutil

import numpy as np
import pytest

from syncmoss import fitting_io, model_io
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.spectrum_parameters import (
    SpectrumParameters, parse_path_box, format_path_box, substitute, names_used,
    missing_names, read_parameters_file, apply_parameters, parameters_file_text,
    write_parameters_file, sequence_problems,
)

from conftest import redirect_params_dir_to_tmp

DELTA = NB + 1      # flat index of the Sextet's delta (row 1, column 1)


# --- the path box ---------------------------------------------------------------

@pytest.mark.parametrize("text, entries", [
    ("['a.dat',\n'b.dat']", [('a.dat', ()), ('b.dat', ())]),
    ("[('a.dat', 4.2, 0), 'b.dat']", [('a.dat', (4.2, 0.0)), ('b.dat', ())]),
    ("('a.dat', 4.2)", [('a.dat', (4.2,))]),
    ("'a.dat', 'b.dat'", [('a.dat', ()), ('b.dat', ())]),
    ("[('a.dat',)]", [('a.dat', ())]),
    ("C:/data/a.dat", [('C:/data/a.dat', ())]),             # not a literal: fallback
    ("a.dat, b.dat", [('a.dat', ()), ('b.dat', ())]),
    ("Model_6", [('Model_6', ())]),
    ("", []),
])
def test_the_path_box_takes_paths_and_tuples(text, entries):
    assert parse_path_box(text) == entries


CAL = r'C:\Users\yaroslav\PycharmProjects\SYNCmoss\syncmoss\syncmoss\parameters\Calibration.dat'


@pytest.mark.parametrize("text, entries", [
    # typed as Windows paths are: \U, \n, \a are NOT Python escapes here
    (rf"[('{CAL}', 0.5), ('{CAL}', 2)]", [(CAL, (0.5,)), (CAL, (2.0,))]),
    (r"['C:\data\new\a.dat']", [(r'C:\data\new\a.dat', ())]),
    (r"C:\data\new\a.dat", [(r'C:\data\new\a.dat', ())]),
    (r"['C:\data\run1\']", [('C:\\data\\run1\\', ())]),                 # a folder
    # typed the Python way, doubled: counted once
    (r"[('C:\\data\\a.dat', 1)]", [(r'C:\data\a.dat', (1.0,))]),
    # a network path keeps its leading \\, either way it was typed
    (r"[('\\server\share\a.dat', 1)]", [(r'\\server\share\a.dat', (1.0,))]),
    (r"['\\\\server\\share\\a.dat']", [(r'\\server\share\a.dat', ())]),
])
def test_backslashes_in_the_path_box_are_path_separators(text, entries):
    assert parse_path_box(text) == entries


@pytest.mark.parametrize("text", [
    rf"['{CAL}', 0.5), ('{CAL}', 2)]",          # the first entry's ( missing
    rf"[('{CAL}', 0.5), ('{CAL}', 2]",          # the last entry's ) missing
    rf"[('{CAL}' 0.5)]",                        # a comma missing
])
def test_a_broken_entry_is_refused_not_cut_into_fragments(text):
    with pytest.raises(ValueError, match="could not be read"):
        parse_path_box(text)


def test_a_bare_path_with_parentheses_is_still_a_path():
    assert parse_path_box(r"C:\data\Fe (2).dat") == [(r'C:\data\Fe (2).dat', ())]


@pytest.mark.gui
def test_a_broken_path_box_is_reported_as_such(physics_app, tmp_path):
    from syncmoss.syncmoss_main import PhysicsApp
    from syncmoss.spectrum_parameters import PATH_BOX_UNREADABLE
    app = physics_app
    app.process_path.setPlainText(rf"['{CAL}', 0.5), ('{CAL}', 2)]")
    assert PhysicsApp.check_spectrum_paths_exist(app.parse_process_path()) == PATH_BOX_UNREADABLE
    assert not app.apply_spectrum_parameters_file(_file(tmp_path, "1 2\n"))
    assert "could not be read" in app.log.toPlainText()


def test_windows_paths_round_trip_through_the_path_box():
    entries = [(CAL, (0.5,)), (r'\\server\share\b.dat', ()), (r"C:\O'Brien\c.dat", (1.0,))]
    text = format_path_box(entries)
    assert text.splitlines()[0] == rf"[('{CAL}', 0.5),"                # as the user types it
    assert parse_path_box(text) == entries


@pytest.mark.parametrize("text", ["[('a.dat', 'x')]", "[('a.dat', True)]", "[4.2]",
                                  "[('a.dat', 1e999)]"])
def test_a_path_box_item_that_is_neither_is_refused(text):
    with pytest.raises(ValueError):
        parse_path_box(text)


def test_plain_paths_are_written_exactly_as_choose_file_writes_them():
    paths = ['C:/data/a.dat', 'C:/data/b.dat']
    choose_file_text = str(paths).replace("', '", "',\n'")      # syncmoss_main.choose_file
    assert format_path_box([(p, ()) for p in paths]) == choose_file_text


def test_tuples_round_trip_through_the_path_box():
    entries = [('a.dat', (4.2, 0.0)), ('b.dat', ()), ('c d.dat', (-1.5, 1e-05))]
    text = format_path_box(entries)
    assert text == "[('a.dat', 4.2, 0),\n'b.dat',\n('c d.dat', -1.5, 1e-05)]"
    assert parse_path_box(text) == entries


# --- the names ------------------------------------------------------------------

def test_the_names_are_replaced_by_the_values_in_parentheses():
    sp = SpectrumParameters(3, (77, -5))
    assert substitute('p[9]+N*2', sp) == 'p[9]+(3)*2'
    assert substitute('p[9]+N1*2', sp) == 'p[9]+(77.0)*2'
    assert eval(substitute('N2**2', sp)) == 25.0         # not -(5**2)


def test_only_whole_names_are_replaced():
    sp = SpectrumParameters(1, tuple(float(k) for k in range(1, 11)))
    assert eval(substitute('N10 - N1', sp)) == 9.0       # N10 is not N1 followed by 0
    text = 'Ns + None + NaN + x_N1 + N_1'
    assert substitute(text, sp) == text
    assert names_used([text]) == []


def test_a_name_without_a_value_is_left_for_the_evaluation_to_report():
    sp = SpectrumParameters(1, (4.2, 0))
    assert substitute('N3 + N0 + N01', sp) == 'N3 + N0 + N01'
    assert missing_names(['p[0] + N + N1 + N3', 'N0'], sp) == ['N0', 'N3']


def test_every_spectrum_of_a_sequence_is_checked():
    entries = [('a.dat', (1.0, 2.0)), ('b.dat', (1.0,)), ('c.dat', ())]
    assert sequence_problems(['p[1] + N2', 'N'], entries) == ['N2 has no value for b.dat, c.dat']
    assert sequence_problems(['p[1] + N'], [('a.dat', ())]) == []


# --- the parameters file --------------------------------------------------------

def _file(tmp_path, text, name='parameters.txt'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_comments_are_skipped_and_spaces_and_tabs_both_separate(tmp_path):
    table = read_parameters_file(_file(tmp_path, "# temperature, K\n4.2 \t 77  150\n\n#N2\n0\t0 45\n"))
    assert table == {'basenames': None, 'numbers': None,
                     'rows': [[4.2, 77.0, 150.0], [0.0, 0.0, 45.0]]}


def test_number_needs_basename(tmp_path):
    with pytest.raises(ValueError, match="'number' is a reserved name and can be used only "
                                         "along with 'basename'"):
        read_parameters_file(_file(tmp_path, "#number\n1 2\n#N1\n4.2 77\n"))


@pytest.mark.parametrize("text, message", [
    ("#N1\n4,2 77\n", "line 2: '4,2' is not a number .*decimal point"),
    ("#N1\n4.2 inf\n", "line 2: 'inf' is not a finite number"),
    ("#basename\n#N1\n4.2\n", "line 2: the line after #basename is missing"),
    ("#basename\na.dat\n#basename\na.dat\n", "#basename appears twice"),
    ("#basename\n", "the line after #basename is missing"),
])
def test_a_bad_file_is_refused_with_the_line(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_parameters_file(_file(tmp_path, text))


def test_by_order_every_line_needs_one_number_per_spectrum(tmp_path):
    entries = [('a.dat', ()), ('b.dat', ()), ('c.dat', ())]
    table = read_parameters_file(_file(tmp_path, "4.2 77 150 300\n0 0 45\n"))
    new_entries, notes = apply_parameters(entries, table)
    assert new_entries == [('a.dat', (4.2, 0.0)), ('b.dat', (77.0, 0.0)), ('c.dat', (150.0, 45.0))]
    assert notes == []

    short = read_parameters_file(_file(tmp_path, "4.2 77 150\n0 0\n"))
    with pytest.raises(ValueError, match="3 spectra need 3 numbers per line"):
        apply_parameters(entries, short)


def test_by_basename_each_spectrum_takes_its_own_column(tmp_path):
    entries = [('C:/x/c.dat', (9.0,)), ('C:/x/a.dat', ()), ('C:/x/b.dat', ())]
    table = read_parameters_file(_file(tmp_path, "#basename\na.dat b.dat c.dat\n#N1\n1 2 3\n"))
    new_entries, notes = apply_parameters(entries, table)
    assert new_entries == [('C:/x/c.dat', (3.0,)), ('C:/x/a.dat', (1.0,)), ('C:/x/b.dat', (2.0,))]
    assert notes == []                              # no #number: the order stays


def test_by_basename_refusals(tmp_path):
    table = read_parameters_file(_file(tmp_path, "#basename\na.dat b.dat\n#N1\n1 2\n"))
    with pytest.raises(ValueError, match="not listed under #basename: c.dat"):
        apply_parameters([('a.dat', ()), ('c.dat', ())], table)
    with pytest.raises(ValueError, match="share the name a.dat"):
        apply_parameters([('300K/a.dat', ()), ('77K/a.dat', ())], table)
    wrong = read_parameters_file(_file(tmp_path, "#basename\na.dat b.dat\n#N1\n1 2 3\n"))
    with pytest.raises(ValueError, match="2 basenames but 3 numbers in N1"):
        apply_parameters([('a.dat', ()), ('b.dat', ())], wrong)


def test_number_reorders_the_spectra(tmp_path):
    entries = [('a.dat', ()), ('b.dat', ()), ('c.dat', ())]
    table = read_parameters_file(_file(
        tmp_path, "#basename\na.dat b.dat c.dat\n#number\n3 1 2\n#N1\n30 10 20\n"))
    new_entries, notes = apply_parameters(entries, table)
    assert new_entries == [('b.dat', (10.0,)), ('c.dat', (20.0,)), ('a.dat', (30.0,))]
    assert notes == ["the spectra were reordered by #number"]


def test_n_stays_the_position_when_the_numbers_are_not_1_to_n(tmp_path):
    table = read_parameters_file(_file(tmp_path, "#basename\na.dat b.dat\n#number\n9 4\n"))
    new_entries, notes = apply_parameters([('a.dat', (5.0,)), ('b.dat', ())], table)
    assert new_entries == [('b.dat', ()), ('a.dat', ())]    # no parameter lines: values go
    assert any("N counts the spectra" in note for note in notes)


def test_the_saved_file_has_basename_number_and_named_lines(tmp_path):
    entries = [('C:/x/Fe 4K.dat', (4.2, 0.0)), ('C:/x/Fe_77K.dat', (77.0, 45.0))]
    text = parameters_file_text(entries)
    assert text.splitlines() == ["#basename", "'Fe 4K.dat'\tFe_77K.dat", "#number", "1\t2",
                                 "#N1", "4.2\t77", "#N2", "0\t45"]
    path = _file(tmp_path, '')
    write_parameters_file(path, entries)
    table = read_parameters_file(path)
    assert table['basenames'] == ['Fe 4K.dat', 'Fe_77K.dat']
    reversed_box = [(p, ()) for p, _ in reversed(entries)]
    assert apply_parameters(reversed_box, table) == (entries, ["the spectra were reordered by #number"])


def test_a_file_is_saved_only_when_it_can_be_read_back():
    with pytest.raises(ValueError, match="same number of parameters"):
        parameters_file_text([('a.dat', (1.0,)), ('b.dat', ())])
    with pytest.raises(ValueError, match="share the name"):
        parameters_file_text([('x/a.dat', ()), ('y/a.dat', ())])


# --- through the application ------------------------------------------------------

class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _linked_delta(app, tmp_path, text):
    """Sextet with its delta linked to an Expression row holding *text*; returns its slot."""
    redirect_params_dir_to_tmp(app, tmp_path)
    pt = app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, 'Expression')
    model_io._set_row_param(pt.row_widgets[2], 0, text, '', '', 'False')
    slot = int(model_io.read_model(app, substitute_names=False)[8][0])
    model_io._set_row_param(pt.row_widgets[1], 1, f'=[{slot},1]', '', '', 'False')
    return slot


@pytest.mark.gui
def test_read_model_uses_the_first_spectrum_of_the_path_box(physics_app, tmp_path):
    app = physics_app
    _linked_delta(app, tmp_path, 'p[0]*0 + N1*2 + N')
    app.process_path.setPlainText("[('a.dat', 77), ('b.dat', 80)]")
    assert model_io.read_model(app)[7] == ['p[0]*0 + (77.0)*2 + (1)']
    assert model_io.read_model(app, substitute_names=False)[7] == ['p[0]*0 + N1*2 + N']
    second = SpectrumParameters(2, (80,))
    assert model_io.read_model(app, spectrum_parameters=second)[7] == ['p[0]*0 + (80.0)*2 + (2)']


@pytest.mark.gui
def test_a_missing_value_is_reported_before_the_start(physics_app, tmp_path):
    app = physics_app
    _linked_delta(app, tmp_path, 'N1 + N2')
    app.process_path.setPlainText("[('a.dat', 0.1)]")
    problems = model_io.validate_user_expressions(app)
    assert len(problems) == 1
    assert problems[0]['text'] == 'N1 + N2'                  # as typed
    assert "N2 has no value for a.dat" in str(problems[0]['error'])


@pytest.mark.gui
def test_the_names_are_refused_with_nbaseline(physics_app, tmp_path):
    app = physics_app
    _linked_delta(app, tmp_path, 'N')
    app.params_table.select_model(3, 'Nbaseline')
    problems = model_io.validate_user_expressions(app)
    assert len(problems) == 1
    assert "not in a model with Nbaseline" in str(problems[0]['error'])


@pytest.mark.gui
def test_loading_a_file_rewrites_the_path_box(physics_app, tmp_path):
    app = physics_app
    app.process_path.setPlainText("['C:/x/a.dat',\n'C:/x/b.dat']")
    good = _file(tmp_path, "#basename\nb.dat a.dat\n#number\n1 2\n#N1\n20 10\n", 'good.txt')
    assert app.apply_spectrum_parameters_file(good)
    assert app.process_path.toPlainText() == "[('C:/x/b.dat', 20),\n('C:/x/a.dat', 10)]"
    assert app.parse_process_path() == ['C:/x/b.dat', 'C:/x/a.dat']
    assert "reordered by #number" in app.log.toPlainText()

    bad = _file(tmp_path, "#number\n1 2\n#N1\n1 2\n", 'bad.txt')
    before = app.process_path.toPlainText()
    assert not app.apply_spectrum_parameters_file(bad)
    assert app.process_path.toPlainText() == before           # nothing changed
    assert "reserved name" in app.log.toPlainText()


@pytest.mark.gui
def test_a_sequence_spectrum_is_fitted_with_its_own_values(physics_app, tmp_path, monkeypatch):
    """N = 2 and N1 = 0.25 reach the start vector of that spectrum's fit."""
    app = physics_app
    slot = _linked_delta(app, tmp_path, 'N1 + 0.1*N')
    app.initialize_parameters()
    received = {}

    class Captured(Exception):
        pass

    def capture(func, x, y, p0, **kwargs):
        received['p0'] = np.array(p0, dtype=float)
        raise Captured()

    monkeypatch.setattr(fitting_io.mi, 'minimi_hi', capture)
    fitting_io.fit_single_spectrum(
        app, app.calibration_path, _SerialPool(),
        spectrum_parameters=SpectrumParameters(2, (0.25,), app.calibration_path))
    assert received['p0'][slot] == pytest.approx(0.45)
    assert received['p0'][DELTA] == pytest.approx(0.45)


@pytest.mark.gui
def test_a_sequence_records_the_values_it_used(physics_app, tmp_path):
    """Two spectra fitted in sequence, delta forced to N1: the run's parameters
    file, the N/N1 columns of each _param.txt and the fitted delta."""
    app = physics_app
    _linked_delta(app, tmp_path, 'N1')
    row = app.params_table.row_widgets[1]
    for col in range(14):                        # only the baseline Ns stays free
        if col != 1:
            row.layout().itemAt(col + 1).widget().layout().itemAt(0).layout().itemAt(1).widget().setChecked(True)
    app.initialize_parameters()
    files = []
    for name in ('fe_a.dat', 'fe_b.dat'):
        files.append(os.path.join(str(tmp_path), name))
        shutil.copy2(app.calibration_path, files[-1])
    entries = [(files[0], (0.1,)), (files[1], (0.3,))]
    app.process_path.setPlainText(format_path_box(entries))
    save_base = os.path.join(str(tmp_path), 'result', 'result')
    app.save_path.setText(save_base)
    app.pool = _SerialPool()
    app.inprogress = True
    delta_name = app.params_table.get_parameter_names()[1][1]
    os.makedirs(os.path.dirname(save_base))
    with open(save_base + '_param.txt', 'w', encoding='utf-8') as f:
        f.write('#File\tan older run\nold.dat\t1\n')    # kept: the run extends the file

    app.start_sequential_fitting(files, entries)
    assert app.sequential_fitting_thread.wait(120000)
    from PySide6.QtWidgets import QApplication
    for _ in range(20):                          # deliver the queued per-spectrum results
        QApplication.processEvents()

    table = read_parameters_file(save_base + '_inputs.txt')
    assert table == {'basenames': ['fe_a.dat', 'fe_b.dat'], 'numbers': [1.0, 2.0], 'rows': [[0.1, 0.3]]}

    # ONE _param.txt: the older run, then this run's names and a line per spectrum
    with open(save_base + '_param.txt', encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert len(lines) == 5 and lines[:2] == ['#File\tan older run', 'old.dat\t1']
    header = lines[2].split('\t')
    assert header[:3] == ['#File', 'N', 'N1']
    for row_text, (name, number, value) in zip(lines[3:], (('fe_a', 1, 0.1), ('fe_b', 2, 0.3))):
        row_values = row_text.split('\t')
        assert row_values[0] == name + '.dat'
        assert int(row_values[1]) == number and float(row_values[2]) == value
        assert float(row_values[header.index(delta_name)]) == pytest.approx(value)   # forced
        assert not os.path.exists(os.path.join(str(tmp_path), 'result', name + '_param.txt'))

    # ONE HTML page with the pictures of every spectrum
    with open(save_base + '_result_table_PNG.html', encoding='utf-8') as f:
        page = f.read()
    assert page.startswith('<!DOCTYPE html>') and page.rstrip().endswith('</html>')
    assert '<h3>1. fe_a.dat: N1 = 0.1</h3>' in page and '<h3>2. fe_b.dat: N1 = 0.3</h3>' in page
    assert page.count('<img src="data:image/png;base64,') == 2      # a figure + table each
