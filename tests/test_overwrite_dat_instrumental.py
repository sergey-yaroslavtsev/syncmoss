"""Instrumental function -> Overwrite instrumental function in selected .dat files.

The instrumental function in memory -- for the CMS/SMS mode selected, the lines
RAW -> .dat conversion writes (build_dat_metadata_lines) -- replaces the #@
lines of the selected .dat files, or is added to them. Pinned here:

* the header is every line above the first data line (one that starts with a
  number), however long: the readers used to stop after 40 lines;
* the rewrite: old lines replaced where they were, new ones added just above
  the data (so a calibration's '# sin n1 n2' stays first), every other byte and
  line ending kept, an up-to-date file left alone, a file without data refused;
* the action: only .dat files, never the calibration file whatever its
  spelling, a confirmation first, refused while a calculation runs.

Every file is a temp copy; nothing here touches the user's data or parameters.
"""
import os
import stat

import pytest

import syncmoss.instrumental_io as iio
from syncmoss.instrumental_io import (
    is_data_line, parse_dat_instrumental_metadata, read_dat_metadata_lines,
    write_dat_instrumental_lines, resolve_instrumental_for_file, build_dat_metadata_lines,
)

pytestmark = [pytest.mark.gui]

GCMS_LINES = ["#@GCMS 0.123"]
SMS_LINES = ["#@INSexp 0.2 0.1 0.7 0.09 -0.06 0.43 -0.54 0.013 0.57", "#@INSint 2.0 0.15"]
DATA = b"-1.0\t100.0\n0.0\t90.0\n1.0\t100.0\n"


def _lines(lines, ending=b"\n"):
    return b"".join(line.encode() + ending for line in lines)


# ---------------------------------------------------------------------------
# Where the header ends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line, data", [
    ("-10.5\t12345", True), ("1e3 5", True), ("+0.25 7", True), (".5 1", True),
    ("# comment", False), ("#@GCMS 0.1", False), ("< note", False),
    ("Velocity Counts", False),
])
def test_a_data_line_starts_with_a_number(line, data):
    assert is_data_line(line) is data


def test_the_whole_header_is_read_however_long(tmp_path):
    dat = tmp_path / "long.dat"
    header = "".join(f"# note {i}\n\n" for i in range(60)) + "Velocity Counts\n#@GCMS 0.123\n"
    dat.write_bytes(header.encode() + DATA)
    assert parse_dat_instrumental_metadata(str(dat))['GCMS'] == pytest.approx(0.123)
    assert read_dat_metadata_lines(str(dat)) == GCMS_LINES


def test_a_line_below_the_first_data_line_is_not_header(tmp_path):
    dat = tmp_path / "late.dat"
    dat.write_bytes(DATA + _lines(GCMS_LINES) + DATA)
    assert parse_dat_instrumental_metadata(str(dat))['has_gcms'] is False
    assert read_dat_metadata_lines(str(dat)) == []


# ---------------------------------------------------------------------------
# The rewrite of one file
# ---------------------------------------------------------------------------

def test_old_lines_are_replaced_where_they_were(tmp_path):
    dat = tmp_path / "sms.dat"
    dat.write_bytes(b"# Converted by SYNCMoss\n" + _lines(SMS_LINES) + b"#@INSth 1 2 3\n"
                    + b"# Excluded regions, mm/s: -1..1\n" + DATA)
    assert write_dat_instrumental_lines(str(dat), GCMS_LINES) is True
    assert dat.read_bytes() == (b"# Converted by SYNCMoss\n" + _lines(GCMS_LINES)
                                + b"# Excluded regions, mm/s: -1..1\n" + DATA)


def test_new_lines_go_just_above_the_data(tmp_path):
    """The first line of a calibration file stays first."""
    dat = tmp_path / "with_header.dat"
    dat.write_bytes(b"#\tsin \t12\t1012\n# a note\n\n" + DATA)
    assert write_dat_instrumental_lines(str(dat), SMS_LINES) is True
    assert dat.read_bytes() == b"#\tsin \t12\t1012\n# a note\n\n" + _lines(SMS_LINES) + DATA

    bare = tmp_path / "bare.dat"
    bare.write_bytes(DATA)
    write_dat_instrumental_lines(str(bare), GCMS_LINES)
    assert bare.read_bytes() == _lines(GCMS_LINES) + DATA


def test_every_other_byte_and_line_ending_is_kept(tmp_path):
    """CRLF, bytes that are not ASCII (\\x85 is a line break only to
    str.splitlines), trailing blanks and a last line without its end."""
    dat = tmp_path / "crlf.dat"
    dat.write_bytes(b"# T = 4.2 K \xb5 \x85 note \r\n#@GCMS 0.5\r\n  \r\n"
                    b"-1.0\t100.0\r\n0.0\t90.0  \r\n1.0\t100.0")
    write_dat_instrumental_lines(str(dat), SMS_LINES)
    assert dat.read_bytes() == (b"# T = 4.2 K \xb5 \x85 note \r\n" + _lines(SMS_LINES, b"\r\n")
                                + b"  \r\n-1.0\t100.0\r\n0.0\t90.0  \r\n1.0\t100.0")
    assert [f for f in os.listdir(tmp_path) if f.endswith('.tmp')] == []


def test_a_file_that_already_has_them_is_left_alone(tmp_path):
    dat = tmp_path / "same.dat"
    dat.write_bytes(b"# note\n" + _lines(GCMS_LINES) + DATA)
    before = os.stat(dat).st_mtime_ns
    assert write_dat_instrumental_lines(str(dat), GCMS_LINES) is False
    assert os.stat(dat).st_mtime_ns == before


def test_a_file_without_data_is_refused(tmp_path):
    dat = tmp_path / "comments.dat"
    dat.write_bytes(b"# only a comment\n")
    with pytest.raises(ValueError):
        write_dat_instrumental_lines(str(dat), GCMS_LINES)
    assert dat.read_bytes() == b"# only a comment\n"


def test_a_read_only_file_is_refused(tmp_path):
    dat = tmp_path / "locked.dat"
    dat.write_bytes(DATA)
    os.chmod(dat, stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            write_dat_instrumental_lines(str(dat), GCMS_LINES)
    finally:
        os.chmod(dat, stat.S_IREAD | stat.S_IWRITE)
    assert dat.read_bytes() == DATA


# ---------------------------------------------------------------------------
# The menu action
# ---------------------------------------------------------------------------

def _select(app, paths):
    app.process_path.setPlainText(repr([str(p) for p in paths]))


def _answer(monkeypatch, yes):
    """Answer the confirmation (True: Overwrite); returns what it asked."""
    asked = []
    monkeypatch.setattr(iio, "_confirm_overwrite",
                        lambda app, *texts: asked.append(texts) or yes)
    return asked


def test_the_menu_offers_it(physics_app):
    texts = [a.text() for a in physics_app.instrumental_menu.actions()]
    assert "Overwrite instrumental function in selected .dat files" in texts


@pytest.mark.parametrize("click, written", [("Overwrite", True), ("Cancel", False), (None, False)])
def test_the_question_writes_only_on_overwrite(physics_app, monkeypatch, click, written):
    """The real dialog, its exec replaced by a click (None: closed unanswered)."""
    from PySide6.QtWidgets import QMessageBox
    shown = []

    def exec_(box):
        shown.append((box.text(), box.informativeText(), box.detailedText(),
                      box.defaultButton().text()))
        for button in box.buttons():
            if click and button.text().replace('&', '') == click:
                button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    assert iio._confirm_overwrite(physics_app, "Write?", "notes", "a.dat") is written
    assert shown[0][:3] == ("Write?", "notes", "a.dat")
    assert shown[0][3].replace('&', '') == "Cancel"


def test_only_the_dat_files_get_the_one_in_memory(physics_app, tmp_path, monkeypatch):
    sms = tmp_path / "sms.dat"
    sms.write_bytes(b"# Converted by SYNCMoss\n" + _lines(SMS_LINES) + DATA)
    plain = tmp_path / "plain.dat"
    plain.write_bytes(DATA)
    raw = tmp_path / "raw.mca"
    raw.write_bytes(b"@A 1 2 3\n")
    asked = _answer(monkeypatch, True)
    physics_app.MS_fit.setChecked(True)
    physics_app.GCMS_input.setText("0.097")
    try:
        _select(physics_app, [sms, raw, plain])
        physics_app.overwrite_dat_instrumental_pressed()
        for dat in (sms, plain):
            assert read_dat_metadata_lines(str(dat)) == ["#@GCMS 0.097"]
            assert resolve_instrumental_for_file(physics_app, str(dat))['method'] == 'CMS'
        assert raw.read_bytes() == b"@A 1 2 3\n"
        assert len(asked) == 1 and "into 2 .dat file(s)" in asked[0][0]
        log = physics_app.log.toPlainText()
        assert "written into 2 .dat file(s)" in log
        assert "1 file(s) that are not .dat" in log
    finally:
        physics_app.SMS_fit.setChecked(True)


def test_a_cms_file_becomes_sms(physics_app, tmp_path, monkeypatch):
    """Every old line goes: a #@GCMS left behind would still win."""
    dat = tmp_path / "cms.dat"
    dat.write_bytes(_lines(GCMS_LINES) + DATA)
    _answer(monkeypatch, True)
    physics_app.SMS_fit.setChecked(True)
    _select(physics_app, [dat])
    physics_app.overwrite_dat_instrumental_pressed()
    lines, method = build_dat_metadata_lines(physics_app)
    assert method == 'SMS'
    assert read_dat_metadata_lines(str(dat)) == lines
    resolved = resolve_instrumental_for_file(physics_app, str(dat))
    assert resolved['method'] == 'SMS' and resolved['source'] == 'file'


@pytest.mark.parametrize("box", ["mca", "Model_6", "folder"])
def test_without_a_dat_file_nothing_is_asked(physics_app, tmp_path, monkeypatch, box):
    asked = _answer(monkeypatch, True)
    physics_app.process_path.setPlainText({
        "mca": repr([str(tmp_path / "raw.mca")]),
        "Model_6": "Model_6",
        "folder": repr([str(tmp_path) + os.sep]),
    }[box])
    physics_app.overwrite_dat_instrumental_pressed()
    assert asked == []
    assert physics_app.log.toPlainText() == \
        "Instrumental function information can be written only in .dat files"


def test_declining_writes_nothing(physics_app, tmp_path, monkeypatch):
    dat = tmp_path / "plain.dat"
    dat.write_bytes(DATA)
    _answer(monkeypatch, False)
    _select(physics_app, [dat])
    physics_app.overwrite_dat_instrumental_pressed()
    assert dat.read_bytes() == DATA
    assert "canceled" in physics_app.log.toPlainText()


def test_the_calibration_file_is_never_written(physics_app, tmp_path, monkeypatch):
    """Calibration.dat of the parameters folder and the calibration in use, in
    whatever spelling; a file of that name elsewhere is a spectrum like any other."""
    params_calibration = os.path.join(physics_app.params_dir, "Calibration.dat")
    in_use = physics_app.calibration_path
    spellings = [params_calibration, params_calibration.replace("\\", "/"), in_use,
                 os.path.join(physics_app.params_dir, "..", "parameters", "Calibration.dat")]
    if os.name == 'nt':
        spellings.append(params_calibration.upper())
    try:
        spellings.append(os.path.relpath(params_calibration))
    except ValueError:      # another drive than the working folder
        pass
    before = {path: open(path, 'rb').read() for path in (params_calibration, in_use)}
    asked = _answer(monkeypatch, True)
    for spelling in spellings:
        _select(physics_app, [spelling])
        physics_app.overwrite_dat_instrumental_pressed()
        assert "(the calibration file)" in physics_app.log.toPlainText(), spelling
    assert asked == []
    for path, content in before.items():
        assert open(path, 'rb').read() == content

    elsewhere = tmp_path / "beamtime" / "Calibration.dat"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(DATA)
    _select(physics_app, [params_calibration, elsewhere])
    physics_app.overwrite_dat_instrumental_pressed()
    assert read_dat_metadata_lines(str(elsewhere)) == build_dat_metadata_lines(physics_app)[0]
    assert "Not written: Calibration.dat (the calibration file)" in physics_app.log.toPlainText()


def test_refused_while_a_calculation_runs(physics_app, tmp_path, monkeypatch):
    dat = tmp_path / "plain.dat"
    dat.write_bytes(DATA)
    asked = _answer(monkeypatch, True)
    _select(physics_app, [dat])
    physics_app.inprogress = True
    physics_app.busy_with = 'Fitting'
    try:
        physics_app.overwrite_dat_instrumental_pressed()
    finally:
        physics_app.inprogress = False
    assert asked == [] and dat.read_bytes() == DATA
    assert "not available right now" in physics_app.log.toPlainText()
