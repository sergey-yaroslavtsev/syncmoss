"""Per-spectrum instrumental metadata tests (#@GCMS / #@INSexp / #@INSint).

Covers the CMS/SMS metadata feature end to end:

* parsing the #@GCMS header (and its precedence over #@INSexp/#@INSint),
* RAW->DAT conversion embedding #@GCMS in CMS mode and #@INSexp/#@INSint in
  SMS mode,
* resolve_instrumental_for_file() for every "from file" / internal combination,
* real fits of Calibration.dat saved in multiple ways: a mixed CMS+SMS
  simultaneous fit and a two-SMS fit whose files carry different metadata.
"""
import os
import shutil

import numpy as np
import pytest

from syncmoss.instrumental_io import (
    parse_dat_instrumental_metadata,
    resolve_instrumental_for_file,
    analyze_instrumental_methods,
    get_sms_instrumental_from_global_files,
    same_method_params,
    read_dat_metadata_lines,
    dat_metadata_key,
    shared_dat_metadata_lines,
)

pytestmark = [pytest.mark.gui]

# A second, deliberately different SMS instrumental function (x0 and MulCo
# differ from the global INSint.txt) to prove per-spectrum resolution.
ALT_INSEXP = "0.2 0.1 0.7 0.09 -0.06 0.43 -0.54 0.013 0.57"
ALT_INSINT = "2.0 0.15"

RAW_SAMPLE = os.path.join(os.path.dirname(__file__), "..", "for tests", "000.mca")


def write_dat_variant(directory, name, source_dat, header_lines):
    """Save a copy of an existing two-column .dat spectrum with the given
    instrumental header lines — the same layout RAW->DAT conversion produces."""
    dst = os.path.join(str(directory), name)
    with open(source_dat) as src, open(dst, "w") as out:
        out.write("# Converted by SYNCMoss\n")
        for line in header_lines:
            out.write(line + "\n")
        for line in src:
            if line.strip() and not line.lstrip().startswith(("#", "<")):
                out.write(line)
    return dst


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def test_parse_gcms_metadata(tmp_path):
    dat = tmp_path / "cms.dat"
    dat.write_text("# Converted by SYNCMoss\n#@GCMS 0.123\n0.0\t100.0\n1.0\t99.0\n")
    meta = parse_dat_instrumental_metadata(str(dat))
    assert meta["has_gcms"] is True
    assert meta["GCMS"] == pytest.approx(0.123)
    assert meta["has_insexp"] is False
    assert meta["has_insint"] is False


def test_parse_sms_metadata(tmp_path):
    dat = tmp_path / "sms.dat"
    dat.write_text(
        f"# Converted by SYNCMoss\n#@INSexp {ALT_INSEXP}\n#@INSint {ALT_INSINT}\n"
        "0.0\t100.0\n1.0\t99.0\n"
    )
    meta = parse_dat_instrumental_metadata(str(dat))
    assert meta["has_gcms"] is False
    assert meta["has_insexp"] is True
    assert meta["has_insint"] is True
    assert np.allclose(meta["INS"], [float(v) for v in ALT_INSEXP.split()])
    assert meta["MulCo"] == pytest.approx(2.0)
    assert meta["x0"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# resolve_instrumental_for_file combinations
# ---------------------------------------------------------------------------

def test_resolve_prefers_file_metadata(physics_app, tmp_path):
    """With the "from .dat file" option enabled, the file's own metadata decides
    the method — even when the UI checkbox says otherwise."""
    gcms_dat = write_dat_variant(tmp_path, "cms.dat", physics_app.calibration_path, ["#@GCMS 0.123"])
    sms_dat = write_dat_variant(
        tmp_path, "sms.dat", physics_app.calibration_path,
        [f"#@INSexp {ALT_INSEXP}", f"#@INSint {ALT_INSINT}"],
    )

    physics_app.SMS_fit.setChecked(True)  # UI says SMS ...
    resolved = resolve_instrumental_for_file(physics_app, gcms_dat, use_dat_metadata=True)
    assert resolved["method"] == "CMS"  # ... but the file says CMS
    assert resolved["Met"] == 1
    assert resolved["source"] == "file"
    assert resolved["INS"] == pytest.approx(0.123)
    assert resolved["x0"] == 0.0
    assert "CMS" in resolved["note"] and "#@GCMS" in resolved["note"]

    physics_app.MS_fit.setChecked(True)  # UI says CMS ...
    resolved = resolve_instrumental_for_file(physics_app, sms_dat, use_dat_metadata=True)
    assert resolved["method"] == "SMS"  # ... but the file says SMS
    assert resolved["Met"] == 0
    assert resolved["source"] == "file"
    assert resolved["MulCo"] == pytest.approx(2.0)
    assert resolved["x0"] == pytest.approx(0.15)
    physics_app.SMS_fit.setChecked(True)


def test_resolve_internal_when_disabled(physics_app, tmp_path):
    """With the option disabled, the UI method with internal values is used."""
    gcms_dat = write_dat_variant(tmp_path, "cms.dat", physics_app.calibration_path, ["#@GCMS 0.123"])

    physics_app.SMS_fit.setChecked(True)
    resolved = resolve_instrumental_for_file(physics_app, gcms_dat, use_dat_metadata=False)
    assert resolved["method"] == "SMS"
    assert resolved["source"] == "internal"
    INS_global, MulCo_global, x0_global = get_sms_instrumental_from_global_files(physics_app)
    assert np.allclose(np.atleast_1d(resolved["INS"]), np.atleast_1d(INS_global))
    assert resolved["MulCo"] == pytest.approx(MulCo_global)
    assert resolved["x0"] == pytest.approx(x0_global)

    physics_app.MS_fit.setChecked(True)
    physics_app.GCMS_input.setText("0.2")
    resolved = resolve_instrumental_for_file(physics_app, gcms_dat, use_dat_metadata=False)
    assert resolved["method"] == "CMS"
    assert resolved["source"] == "internal"
    assert resolved["INS"] == pytest.approx(0.2)  # the G input field, not the file
    physics_app.GCMS_input.setText("0.097")
    physics_app.SMS_fit.setChecked(True)


def test_resolve_fallback_without_metadata(physics_app, tmp_path):
    """A .dat without any instrumental header falls back to the internal values
    of the UI-selected method, and says why."""
    plain_dat = write_dat_variant(tmp_path, "plain.dat", physics_app.calibration_path, [])
    physics_app.SMS_fit.setChecked(True)
    resolved = resolve_instrumental_for_file(physics_app, plain_dat, use_dat_metadata=True)
    assert resolved["method"] == "SMS"
    assert resolved["source"] == "internal"
    assert "no usable .dat metadata" in resolved["note"]


def test_gcms_wins_when_file_has_both(physics_app, tmp_path):
    """#@GCMS marks a CMS spectrum even if SMS metadata is also present."""
    both_dat = write_dat_variant(
        tmp_path, "both.dat", physics_app.calibration_path,
        ["#@GCMS 0.111", f"#@INSexp {ALT_INSEXP}", f"#@INSint {ALT_INSINT}"],
    )
    resolved = resolve_instrumental_for_file(physics_app, both_dat, use_dat_metadata=True)
    assert resolved["method"] == "CMS"
    assert resolved["INS"] == pytest.approx(0.111)


# ---------------------------------------------------------------------------
# RAW -> DAT conversion
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(RAW_SAMPLE), reason="RAW sample file not available")
@pytest.mark.parametrize("mode", ["CMS", "SMS"])
def test_raw_to_dat_embeds_method_metadata(physics_app, tmp_path, mode):
    from syncmoss.syncmoss_main import RawToDatThread

    raw_copy = os.path.join(str(tmp_path), "000.mca")
    shutil.copy2(RAW_SAMPLE, raw_copy)
    out_dir = os.path.join(str(tmp_path), mode) + os.sep
    os.makedirs(out_dir, exist_ok=True)

    physics_app.MS_fit.setChecked(mode == "CMS")  # mutual exclusivity flips SMS
    physics_app.GCMS_input.setText("0.097")

    messages = []
    thread = RawToDatThread(physics_app, [raw_copy], physics_app.calibration_path, out_dir)
    thread.finished.connect(lambda m: messages.append(("ok", m)))
    thread.error.connect(lambda m: messages.append(("err", m)))
    thread.run()  # synchronous: no need to spin the thread

    physics_app.SMS_fit.setChecked(True)

    assert messages and messages[0][0] == "ok", f"conversion failed: {messages}"
    out_file = os.path.join(out_dir, "000.dat")
    assert os.path.exists(out_file)

    meta = parse_dat_instrumental_metadata(out_file)
    if mode == "CMS":
        assert meta["has_gcms"] is True
        assert meta["GCMS"] == pytest.approx(0.097)
        assert meta["has_insexp"] is False and meta["has_insint"] is False
        assert "#@GCMS" in messages[0][1]
    else:
        assert meta["has_gcms"] is False
        assert meta["has_insexp"] is True and meta["has_insint"] is True
        INS_global, MulCo_global, x0_global = get_sms_instrumental_from_global_files(physics_app)
        assert np.allclose(meta["INS"], np.atleast_1d(INS_global))
        assert meta["MulCo"] == pytest.approx(MulCo_global)
        assert meta["x0"] == pytest.approx(x0_global)
        assert "#@INSexp" in messages[0][1]

    # The converted file must still load as a regular two-column spectrum
    from syncmoss.spectrum_io import load_spectrum
    A_list, B_list = load_spectrum(physics_app, [out_file], calibration_path=physics_app.calibration_path)
    assert len(A_list) == 1 and len(A_list[0]) > 0


# ---------------------------------------------------------------------------
# Real fits: Calibration.dat saved in multiple ways, fitted in combinations
# ---------------------------------------------------------------------------

def _prepare_two_spectrum_fit(window, file_a, file_b):
    """Point the GUI at two spectrum files and build a Sextet+Nbaseline+Sextet model."""
    window.jn0_input.setText("16")  # keep the transmission integral cheap
    window.process_path.setPlainText(repr([file_a, file_b]))
    window.path_list = [file_a, file_b]
    window.show_pressed()  # fills baseline Ns from the spectrum background

    pt = window.params_table
    pt.select_model(1, "Sextet")
    pt.select_model(2, "Nbaseline")
    pt.select_model(3, "Sextet")
    assert pt.get_model_list() == ["baseline", "Sextet", "Nbaseline", "Sextet"]
    assert window.initialize_parameters()


def _set_param_value(window, row, col, text):
    """Set the value field of parameter (row, col) in the parameters table."""
    row_widget = window.params_table.row_widgets[row]
    param_widget = row_widget.layout().itemAt(col + 1).widget()
    value_input = param_widget.layout().itemAt(1).widget()
    value_input.setText(text)


@pytest.mark.slow
def test_fit_mixed_cms_and_sms_spectra(physics_app, tmp_path):
    """One CMS spectrum (#@GCMS) and one SMS spectrum (#@INSexp/#@INSint) fitted
    simultaneously; the log note must state the mixed fit explicitly."""
    from multiprocessing.pool import ThreadPool
    import syncmoss.fitting_io as fitting_io

    cal = physics_app.calibration_path
    gcms_dat = write_dat_variant(tmp_path, "cms_spec.dat", cal, ["#@GCMS 0.097"])
    sms_dat = write_dat_variant(
        tmp_path, "sms_spec.dat", cal,
        [f"#@INSexp {ALT_INSEXP}", f"#@INSint {ALT_INSINT}"],
    )

    # The two files must resolve to different methods
    mp_a = resolve_instrumental_for_file(physics_app, gcms_dat, use_dat_metadata=True)
    mp_b = resolve_instrumental_for_file(physics_app, sms_dat, use_dat_metadata=True)
    assert mp_a["method"] == "CMS" and mp_b["method"] == "SMS"
    assert not same_method_params(mp_a, mp_b)

    pool = ThreadPool(processes=2)
    try:
        _prepare_two_spectrum_fit(physics_app, gcms_dat, sms_dat)
        result = fitting_io.fit_single_spectrum(physics_app, gcms_dat, pool)
    finally:
        pool.close()
        pool.join()

    assert result["success"], result.get("message")
    assert result["is_simultaneous"] is True
    assert np.isfinite(result["chi2"])

    note = result["instrumental_note"]
    assert "Mixed-method simultaneous fit" in note
    assert "cms_spec.dat: CMS" in note and "#@GCMS" in note
    assert "sms_spec.dat: SMS" in note and "#@INSexp" in note


@pytest.mark.slow
def test_fit_two_sms_spectra_with_dedicated_metadata(physics_app, tmp_path):
    """Two SMS spectra whose .dat files carry different instrumental metadata:
    each spectrum of the simultaneous fit must use its own values."""
    from multiprocessing.pool import ThreadPool
    import syncmoss.fitting_io as fitting_io

    cal = physics_app.calibration_path
    INS_global, MulCo_global, x0_global = get_sms_instrumental_from_global_files(physics_app)
    default_insexp = " ".join(str(float(v)) for v in np.atleast_1d(INS_global))
    sms_default = write_dat_variant(
        tmp_path, "sms_default.dat", cal,
        [f"#@INSexp {default_insexp}", f"#@INSint {MulCo_global} {x0_global}"],
    )
    sms_alt = write_dat_variant(
        tmp_path, "sms_alt.dat", cal,
        [f"#@INSexp {ALT_INSEXP}", f"#@INSint {ALT_INSINT}"],
    )

    mp_a = resolve_instrumental_for_file(physics_app, sms_default, use_dat_metadata=True)
    mp_b = resolve_instrumental_for_file(physics_app, sms_alt, use_dat_metadata=True)
    assert mp_a["method"] == mp_b["method"] == "SMS"
    assert not same_method_params(mp_a, mp_b)  # dedicated values per spectrum

    pool = ThreadPool(processes=2)
    try:
        _prepare_two_spectrum_fit(physics_app, sms_default, sms_alt)
        result = fitting_io.fit_single_spectrum(physics_app, sms_default, pool)
    finally:
        pool.close()
        pool.join()

    assert result["success"], result.get("message")
    assert result["is_simultaneous"] is True
    assert np.isfinite(result["chi2"])

    note = result["instrumental_note"]
    assert "sms_default.dat: SMS" in note
    assert "sms_alt.dat: SMS" in note
    assert "Mixed-method" not in note  # same method, only the values differ


@pytest.mark.slow
def test_fit_linked_parameter_across_mixed_spectra(physics_app, tmp_path):
    """A parameter of spectrum 2 linked to spectrum 1 (constraint =[idx,mult])
    must be honoured during a simultaneous fit — even with mixed CMS+SMS, where
    each section is computed with its own instrumental settings. This guards the
    concern that per-section computation could break cross-spectrum links.
    """
    from multiprocessing.pool import ThreadPool
    import syncmoss.fitting_io as fitting_io

    cal = physics_app.calibration_path
    cms = write_dat_variant(tmp_path, "cms.dat", cal, ["#@GCMS 0.097"])
    sms = write_dat_variant(
        tmp_path, "sms.dat", cal,
        [f"#@INSexp {ALT_INSEXP}", f"#@INSint {ALT_INSINT}"],
    )

    pool = ThreadPool(processes=2)
    try:
        _prepare_two_spectrum_fit(physics_app, cms, sms)
        # Parameter layout for [baseline(8), Sextet(11), Nbaseline(8), Sextet(11)]:
        #   spectrum-1 Sextet H = p[11], spectrum-2 Sextet H = p[30].
        # Link spectrum-2's H to spectrum-1's H (row 3 = 2nd Sextet, col 3 = H).
        _set_param_value(physics_app, 3, 3, "=[11,1]")
        result = fitting_io.fit_single_spectrum(physics_app, cms, pool)
    finally:
        pool.close()
        pool.join()

    assert result["success"], result.get("message")
    params = result["parameters"]
    # The constraint must hold exactly after the fit
    assert params[30] == pytest.approx(params[11])


# ---------------------------------------------------------------------------
# Metadata preservation across spectrum-processing operations
# ---------------------------------------------------------------------------

def test_save_spectrum_with_metadata_roundtrip(tmp_path):
    from syncmoss.spectrum_io import save_spectrum_with_metadata
    from syncmoss.spectrum_io import load_spectrum

    A = np.linspace(-5.0, 5.0, 12)
    B = 1000.0 - 50.0 * np.exp(-(A ** 2))

    with_meta = os.path.join(str(tmp_path), "with_meta.dat")
    save_spectrum_with_metadata(with_meta, A, B, ["#@GCMS 0.097"])
    assert read_dat_metadata_lines(with_meta) == ["#@GCMS 0.097"]
    assert dat_metadata_key(with_meta) == ("CMS", 0.097)

    without_meta = os.path.join(str(tmp_path), "without_meta.dat")
    save_spectrum_with_metadata(without_meta, A, B, [])
    assert read_dat_metadata_lines(without_meta) == []
    assert dat_metadata_key(without_meta) is None

    # The written data must still load back as a regular two-column spectrum
    A_list, B_list = load_spectrum(None, [with_meta], calibration_path=with_meta)
    assert len(A_list[0]) == len(A)


def test_shared_metadata_keep_clean_and_ignore_missing(physics_app, tmp_path):
    cal = physics_app.calibration_path
    g1 = write_dat_variant(tmp_path, "g1.dat", cal, ["#@GCMS 0.097"])
    g2 = write_dat_variant(tmp_path, "g2.dat", cal, ["#@GCMS 0.097"])
    g_other = write_dat_variant(tmp_path, "g_other.dat", cal, ["#@GCMS 0.200"])
    plain = write_dat_variant(tmp_path, "plain.dat", cal, [])

    # All present metadata agree (a missing one is ignored) -> keep it
    assert shared_dat_metadata_lines([g1, g2, plain]) == ["#@GCMS 0.097"]
    # Conflicting metadata -> cleaned
    assert shared_dat_metadata_lines([g1, g_other]) == []
    # No metadata anywhere -> nothing to keep
    assert shared_dat_metadata_lines([plain]) == []


def test_half_points_preserves_metadata(physics_app, tmp_path):
    from syncmoss.spectrum_io import process_half_spectrum

    src = write_dat_variant(tmp_path, "cms.dat", physics_app.calibration_path, ["#@GCMS 0.097"])
    out = os.path.join(str(tmp_path), "half.dat")
    assert process_half_spectrum(physics_app, src, out, auto_load=False)
    assert dat_metadata_key(out) == ("CMS", 0.097)


def test_sum_keeps_matching_and_drops_conflicting_metadata(physics_app, tmp_path, monkeypatch):
    import syncmoss.spectrum_io as sio

    cal = physics_app.calibration_path
    g1 = write_dat_variant(tmp_path, "s1.dat", cal, ["#@GCMS 0.097"])
    g2 = write_dat_variant(tmp_path, "s2.dat", cal, ["#@GCMS 0.097"])
    g_other = write_dat_variant(tmp_path, "s3.dat", cal, ["#@GCMS 0.200"])

    # sum_all_spectra pops a save dialog and then auto-loads the result; stub both
    monkeypatch.setattr(physics_app, "show_pressed", lambda: None)

    out_same = os.path.join(str(tmp_path), "sum_same.dat")
    monkeypatch.setattr(sio.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (out_same, "")))
    physics_app.process_path.setPlainText(repr([g1, g2]))
    sio.sum_all_spectra(physics_app)
    assert dat_metadata_key(out_same) == ("CMS", 0.097)

    out_conflict = os.path.join(str(tmp_path), "sum_conflict.dat")
    monkeypatch.setattr(sio.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (out_conflict, "")))
    physics_app.process_path.setPlainText(repr([g1, g_other]))
    sio.sum_all_spectra(physics_app)
    assert dat_metadata_key(out_conflict) is None


@pytest.mark.slow
def test_subtract_model_preserves_metadata(physics_app, tmp_path, monkeypatch):
    from multiprocessing.pool import ThreadPool
    import syncmoss.spectrum_io as sio

    sms = write_dat_variant(
        tmp_path, "sms.dat", physics_app.calibration_path,
        [f"#@INSexp {ALT_INSEXP}", f"#@INSint {ALT_INSINT}"],
    )

    physics_app.pool = ThreadPool(processes=2)
    physics_app.SMS_fit.setChecked(True)
    physics_app.jn0_input.setText("16")
    physics_app.process_path.setPlainText(repr([sms]))
    physics_app.path_list = [sms]
    assert physics_app.initialize_parameters()
    physics_app.params_table.select_model(1, "Sextet")

    out = os.path.join(str(tmp_path), "subtracted.dat")
    monkeypatch.setattr(sio.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (out, "")))
    monkeypatch.setattr(physics_app, "show_pressed", lambda: None)
    try:
        sio.subtract_model_from_spectrum(physics_app)
    finally:
        physics_app.pool.close()
        physics_app.pool.join()

    assert os.path.exists(out), physics_app.log.toPlainText()
    # The subtracted spectrum must keep the source's SMS instrumental metadata
    assert dat_metadata_key(out) == dat_metadata_key(sms)


# ---------------------------------------------------------------------------
# Fit-time method resolution and warning dialogs (use_dat on/off, override, mix)
# ---------------------------------------------------------------------------

def _make_cms_and_sms(physics_app, tmp_path):
    cal = physics_app.calibration_path
    cms = write_dat_variant(tmp_path, "cms.dat", cal, ["#@GCMS 0.097"])
    sms = write_dat_variant(
        tmp_path, "sms.dat", cal,
        [f"#@INSexp {ALT_INSEXP}", f"#@INSint {ALT_INSINT}"],
    )
    return cms, sms


def test_analyze_disabled_resolves_uniform_internal(physics_app, tmp_path):
    """Req 1: with .dat metadata disabled, spectra carrying different metadata
    all resolve to the same internal method — nothing to warn about, fit uniform."""
    cms, sms = _make_cms_and_sms(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    overridden, nonuniform, resolved = analyze_instrumental_methods(
        physics_app, [cms, sms], use_dat_metadata=False)
    assert overridden == []
    assert nonuniform is False
    assert {r["method"] for r in resolved} == {"SMS"}
    assert all(r["source"] == "internal" for r in resolved)


def test_analyze_detects_method_override_both_directions(physics_app, tmp_path):
    """Req 3: a file whose metadata method differs from the selected method is
    flagged (CMS-file-in-SMS-mode and SMS-file-in-CMS-mode)."""
    cms, sms = _make_cms_and_sms(physics_app, tmp_path)

    physics_app.SMS_fit.setChecked(True)
    overridden, _, _ = analyze_instrumental_methods(physics_app, [cms], use_dat_metadata=True)
    assert len(overridden) == 1 and overridden[0][1]["method"] == "CMS"

    physics_app.MS_fit.setChecked(True)
    overridden, _, _ = analyze_instrumental_methods(physics_app, [sms], use_dat_metadata=True)
    assert len(overridden) == 1 and overridden[0][1]["method"] == "SMS"

    # Selected method matching the file's metadata -> no override
    overridden, _, _ = analyze_instrumental_methods(physics_app, [cms], use_dat_metadata=True)
    assert overridden == []
    physics_app.SMS_fit.setChecked(True)


def test_analyze_detects_nonuniform(physics_app, tmp_path):
    """Req 4: a CMS+SMS mix and same-method-different-values are both non-uniform;
    identical metadata is uniform."""
    cms, sms = _make_cms_and_sms(physics_app, tmp_path)
    INS_global, MulCo_global, x0_global = get_sms_instrumental_from_global_files(physics_app)
    sms_default = write_dat_variant(
        tmp_path, "sms_def.dat", physics_app.calibration_path,
        [f"#@INSexp {' '.join(str(float(v)) for v in np.atleast_1d(INS_global))}",
         f"#@INSint {MulCo_global} {x0_global}"],
    )
    physics_app.SMS_fit.setChecked(True)

    _, nonuniform, _ = analyze_instrumental_methods(physics_app, [cms, sms], use_dat_metadata=True)
    assert nonuniform is True  # CMS + SMS

    _, nonuniform, _ = analyze_instrumental_methods(physics_app, [sms, sms_default], use_dat_metadata=True)
    assert nonuniform is True  # both SMS, different instrumental function

    _, nonuniform, _ = analyze_instrumental_methods(physics_app, [sms, sms], use_dat_metadata=True)
    assert nonuniform is False  # identical


def test_force_method_sms_ignores_gcms_file(physics_app, tmp_path):
    """force_method='SMS' (used by SMS instrumental refinement) keeps SMS even
    when the file carries #@GCMS — it must return the SMS instrumental array, not
    a scalar G — and honours #@INSexp/#@INSint metadata when present."""
    cms, sms = _make_cms_and_sms(physics_app, tmp_path)

    physics_app.MS_fit.setChecked(True)  # even with CMS selected in the UI ...
    r = resolve_instrumental_for_file(physics_app, cms, use_dat_metadata=True, force_method="SMS")
    assert r["method"] == "SMS" and r["Met"] == 0
    assert len(np.atleast_1d(r["INS"])) >= 2          # an array, not a scalar G
    assert r["source"] == "internal"                  # #@GCMS ignored under force SMS

    r2 = resolve_instrumental_for_file(physics_app, sms, use_dat_metadata=True, force_method="SMS")
    assert r2["method"] == "SMS" and r2["source"] == "file"
    physics_app.SMS_fit.setChecked(True)


def test_force_method_cms_ignores_sms_file(physics_app, tmp_path):
    """force_method='CMS' keeps CMS even when the file carries #@INSexp/#@INSint."""
    cms, sms = _make_cms_and_sms(physics_app, tmp_path)

    physics_app.SMS_fit.setChecked(True)  # even with SMS selected in the UI ...
    r = resolve_instrumental_for_file(physics_app, sms, use_dat_metadata=True, force_method="CMS")
    assert r["method"] == "CMS" and r["Met"] == 1
    assert np.isscalar(r["INS"]) or np.ndim(r["INS"]) == 0   # scalar G
    assert r["source"] == "internal"                          # SMS metadata ignored

    r2 = resolve_instrumental_for_file(physics_app, cms, use_dat_metadata=True, force_method="CMS")
    assert r2["method"] == "CMS" and r2["source"] == "file"


def test_confirm_aborts_when_override_declined(physics_app, tmp_path, monkeypatch):
    """Req 3: declining the method-override warning aborts the fit."""
    cms, _ = _make_cms_and_sms(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)

    seen = []
    def fake_override(overridden, ui_method):
        seen.append((overridden, ui_method))
        return False  # user declines

    monkeypatch.setattr(physics_app, "_warn_method_override", fake_override)
    assert physics_app.confirm_instrumental_methods([cms], "single") is False
    assert len(seen) == 1
    assert seen[0][1] == "SMS"                      # UI method
    assert seen[0][0][0][1]["method"] == "CMS"      # file overrides to CMS


def test_confirm_single_no_mixed_warning(physics_app, tmp_path, monkeypatch):
    """A single spectrum never triggers the multi-spectrum mixed warning."""
    _, sms = _make_cms_and_sms(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)

    mixed_calls = []
    monkeypatch.setattr(physics_app, "_warn_mixed_metadata",
                        lambda resolved, mode: (mixed_calls.append(mode), (True, False))[1])
    # SMS file with SMS selected -> no override, single -> no mixed warning
    assert physics_app.confirm_instrumental_methods([sms], "single") is True
    assert mixed_calls == []


def test_confirm_mixed_warning_and_do_not_ask_again(physics_app, tmp_path, monkeypatch):
    """Req 4: the mixed/different warning shows once; ticking "do not ask again"
    suppresses it for the rest of the session."""
    cms, sms = _make_cms_and_sms(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    monkeypatch.setattr(physics_app, "_warn_method_override", lambda ov, ui: True)

    calls = {"count": 0}
    def fake_mixed(resolved, mode):
        calls["count"] += 1
        return (True, True)  # proceed, and "do not ask again"

    monkeypatch.setattr(physics_app, "_warn_mixed_metadata", fake_mixed)

    assert physics_app.suppress_mixed_metadata_warning is False
    assert physics_app.confirm_instrumental_methods([cms, sms], "simultaneous") is True
    assert calls["count"] == 1
    assert physics_app.suppress_mixed_metadata_warning is True

    # Suppressed now: the popup is not shown again
    assert physics_app.confirm_instrumental_methods([cms, sms], "simultaneous") is True
    assert calls["count"] == 1


def test_confirm_mixed_warning_decline_aborts(physics_app, tmp_path, monkeypatch):
    """Req 4: declining the mixed warning aborts the fit (and does not suppress)."""
    cms, sms = _make_cms_and_sms(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    monkeypatch.setattr(physics_app, "_warn_method_override", lambda ov, ui: True)
    monkeypatch.setattr(physics_app, "_warn_mixed_metadata", lambda resolved, mode: (False, False))
    assert physics_app.confirm_instrumental_methods([cms, sms], "simultaneous") is False
    assert physics_app.suppress_mixed_metadata_warning is False


@pytest.mark.slow
def test_batch_fits_each_spectrum_with_own_metadata(physics_app, tmp_path):
    """Req 2: a batch fits each spectrum on its own, so each picks up its own
    .dat metadata — fitting the CMS file then the SMS file uses CMS then SMS."""
    from multiprocessing.pool import ThreadPool
    import syncmoss.fitting_io as fitting_io

    cms, sms = _make_cms_and_sms(physics_app, tmp_path)
    physics_app.jn0_input.setText("16")
    physics_app.SMS_fit.setChecked(True)  # selected method is the fallback only
    physics_app.process_path.setPlainText(repr([cms]))
    physics_app.path_list = [cms]
    physics_app.show_pressed()
    physics_app.params_table.select_model(1, "Sextet")
    assert physics_app.initialize_parameters()

    pool = ThreadPool(processes=2)
    try:
        r_cms = fitting_io.fit_single_spectrum(physics_app, cms, pool)
        r_sms = fitting_io.fit_single_spectrum(physics_app, sms, pool)
    finally:
        pool.close()
        pool.join()

    assert r_cms["success"] and r_sms["success"]
    assert "CMS" in r_cms["instrumental_note"] and "#@GCMS" in r_cms["instrumental_note"]
    assert "SMS" in r_sms["instrumental_note"] and "#@INSexp" in r_sms["instrumental_note"]
