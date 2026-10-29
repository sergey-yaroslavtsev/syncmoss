"""The 'Spectrum' model row: a measured reference as a component (spectrum_model).

Its line set (T_q, δ_q, G_q of natural-width lines) travels as the row's text in
read_model's Distri list, in model order with the Distr PDF strings. Pinned:

* the transmission integral of a Spectrum equals the product of the Singlets it
  stands for -- T_q scaled to the row's T, δ added to every δ_q, G added in
  quadrature to every G_q -- for CMS and SMS;
* the Distri bookkeeping: a Distr before or after a Spectrum, a Distr ON a
  Spectrum (its copies carry the line set), Nbaseline sections;
* the table row: read_model's layout, the empty line set refused, .mdl round trip;
* the reference fit (spectrum_reference): two Singlets with noise give two
  lines where they are, the number set by the user is reached both ways, an
  absorbing edge is told; the window writes the row on OK.
"""
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss import instrumental_io as iio
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import NAT_WIDTH, SMS_POL_DEFAULT, number_of_baseline_parameters as NB
from syncmoss.spectrum_model import format_spectrum_lines, parse_spectrum_lines

from conftest import FROZEN_PARAMETERS, SerialPool


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_MULCO_SMS, _X0_SMS = _frozen("INSint.txt")
METHODS = {
    "CMS": dict(x0=0.0, MulCo=MulCoCMS, INS=float(_frozen("GCMS.txt", delimiter='\t')), Met=1),
    "SMS": dict(x0=float(_X0_SMS), MulCo=float(_MULCO_SMS),
                INS=np.atleast_1d(_frozen("INSexp.txt")), Met=0),
}
JN = 16
V = np.linspace(-5.0, 5.0, 61)
BASE = [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Three lines: different T_q, positions and widths
LINES = np.array([[0.5, -1.2, 0.0], [1.5, 0.3, 0.25], [1.0, 1.4, 0.6]])
T, D, G = 2.4, 0.15, 0.2


def _ti(p, model, Distri, method, lengths=None, x=V):
    mp = METHODS[method]
    norm = iio.compute_norm(SerialPool(), JN, mp)
    return np.asarray(m5.TI(np.asarray(x, dtype=float), np.asarray(p, dtype=float), list(model),
                            JN, SerialPool(), mp['x0'], mp['MulCo'], mp['INS'], list(Distri), [0],
                            Met=mp['Met'], Norm=norm, lengths=lengths), dtype=float)


def _singlets(lines, t, d, g):
    """The Singlets a Spectrum (t, d, g) with *lines* stands for."""
    share = lines[:, 0] / lines[:, 0].sum()
    return [[t * s, dq + d, NAT_WIDTH, np.hypot(gq, g)] for s, (_, dq, gq) in zip(share, lines)]


def test_parse_and_format_round_trip():
    text = format_spectrum_lines(LINES)
    assert np.array_equal(parse_spectrum_lines(text), LINES)
    for bad in ('', '1,2', '1,2,x', 'nan,0,0'):
        assert parse_spectrum_lines(bad).shape == (0, 3)


@pytest.mark.parametrize("method", ["CMS", "SMS"])
def test_spectrum_is_the_product_of_its_singlets(method):
    text = format_spectrum_lines(LINES)
    got = _ti(BASE + [T, D, G, 0.0], ['Spectrum'], [text], method)
    singlets = _singlets(LINES, T, D, G)
    want = _ti(BASE + sum(singlets, []), ['Singlet'] * len(singlets), [], method)
    assert np.allclose(got, want, rtol=1e-12, atol=0)
    assert np.min(got) < 0.9 * BASE[0]               # not vacuous: it absorbs


@pytest.mark.parametrize("method", ["CMS", "SMS"])
def test_distri_order_with_distributions(method):
    """A Distr before the Spectrum, and one on it -- over δ and over G: the
    same as with the Singlet of a one-line Spectrum in its place."""
    one = np.array([[0.7, 0.0, 0.0]])
    text = format_spectrum_lines(one)
    singlet = _singlets(one, T, D, G)[0]
    distr = [1.0, -0.5, 0.5, 7.0, 0.0]               # par = δ, L, R, Num, placeholder
    pdf = 'exp(-X**2/0.1)'
    cases = [
        (['Singlet', 'Distr', 'Spectrum'], [pdf, text], [pdf]),
        (['Spectrum', 'Distr', 'Singlet'], [text, pdf], [pdf]),
    ]
    sext = [0.8, -0.9, NAT_WIDTH, 0.1]
    for model, distri, distri_ref in cases:
        if model[0] == 'Singlet':
            p = BASE + sext + distr + [T, D, G, 0.0]
            ref = BASE + sext + distr + singlet
        else:
            p = BASE + [T, D, G, 0.0] + distr + sext
            ref = BASE + singlet + distr + sext
        got = _ti(p, model, distri, method)
        want = _ti(ref, [name if name != 'Spectrum' else 'Singlet' for name in model], distri_ref, method)
        assert np.allclose(got, want, rtol=1e-12, atol=0), model
    # the Distr really spreads the Spectrum: not the same as without it
    plain = _ti(BASE + [T, D, G, 0.0] + sext, ['Spectrum', 'Singlet'], [text], method)
    spread = _ti(BASE + [T, D, G, 0.0] + distr + sext, ['Spectrum', 'Distr', 'Singlet'], [text, pdf], method)
    assert np.max(np.abs(plain - spread)) > 1.0
    # over G: par 2 of the Spectrum, par 3 of the Singlet (G_q = 0: the line's G is the row's)
    over_g = [0.05, 0.6, 7.0, 0.0]
    got = _ti(BASE + [T, D, G, 0.0] + [2.0] + over_g, ['Spectrum', 'Distr'], [text, pdf], method)
    want = _ti(BASE + singlet + [3.0] + over_g, ['Singlet', 'Distr'], [pdf], method)
    assert np.allclose(got, want, rtol=1e-12, atol=0)
    assert np.max(np.abs(got - _ti(BASE + [T, D, G, 0.0], ['Spectrum'], [text], method))) > 1.0


def test_nbaseline_sections_read_their_own_line_sets():
    a, b = LINES[:2], LINES[1:]
    x = np.concatenate([V, V])
    p = BASE + [T, D, G, 0.0] + BASE + [1.3, -D, 0.0, 0.0]
    got = _ti(p, ['Spectrum', 'Nbaseline', 'Spectrum'],
              [format_spectrum_lines(a), format_spectrum_lines(b)], 'SMS', lengths=[len(V), len(V)], x=x)
    first = _singlets(a, T, D, G)
    second = _singlets(b, 1.3, -D, 0.0)
    want = _ti(BASE + sum(first, []) + BASE + sum(second, []),
               ['Singlet'] * len(first) + ['Nbaseline'] + ['Singlet'] * len(second), [], 'SMS',
               lengths=[len(V), len(V)], x=x)
    assert np.allclose(got, want, rtol=1e-12, atol=0)


# --- the reference fit ------------------------------------------------------------

REF_X = np.linspace(-4.0, 4.0, 128)
# Two Singlets: T, δ, L, G
REF_TRUTH = [[1.0, -1.0, NAT_WIDTH, 0.1], [0.6, 0.8, NAT_WIDTH, 0.3]]


def _smearing(method, x):
    """A source smeared every way TI takes: the channel average (and, CMS, the
    cosine smearing)."""
    mp = METHODS[method]
    return m5.source_smearing(x, JN, mp['MulCo'], mp['INS'], mp['Met'], 0.2 if method == 'CMS' else 0.0, 0.0,
                              0.25, channel_widths=np.full(len(x), x[1] - x[0]), x0=mp['x0'])


def _lines_model(method='SMS', smearing=None, x=REF_X):
    from syncmoss.spectrum_reference import LinesModel
    mp = METHODS[method]
    norm = (iio.compute_norm(SerialPool(), JN, mp) if smearing is None
            else iio.compute_norm(SerialPool(), JN, mp, x, smearing))
    return LinesModel(x, JN, mp, norm, smearing, SMS_POL_DEFAULT), norm


def _reference(truth, method='SMS', seed=5, fraction=None):
    """Counts of the Singlets *truth* with the baseline of *method*'s reference
    (Nnr/Ns its start, or *fraction*)."""
    from syncmoss.spectrum_reference import nnr_start, with_links
    fraction = nnr_start(method) if fraction is None else fraction
    base = with_links([1e6, 0.0, -1.0, 0.2, 0.0, 0.0, 0.0, 0.0, fraction])
    mp = METHODS[method]
    counts = np.asarray(m5.TI(REF_X, np.concatenate([base, sum(truth, [])]), ['Singlet'] * len(truth), JN,
                              SerialPool(), mp['x0'], mp['MulCo'], mp['INS'], [0], [0], Met=mp['Met'],
                              Norm=_lines_model(method)[1]), float)
    return np.random.default_rng(seed).poisson(counts).astype(float)


@pytest.mark.parametrize("method", ["CMS", "SMS"])
@pytest.mark.parametrize("smeared", [False, True])
def test_lines_model_is_ti(method, smeared):
    """The node sum prepared once (models.node_parts) gives TI's counts, and a
    Jacobian's vectors (counts_of_many, a moved line computed alone) each its
    own; Interrupt stops it."""
    from syncmoss.spectrum_reference import with_links
    smearing = _smearing(method, REF_X) if smeared else None
    model, norm = _lines_model(method, smearing)
    mp = METHODS[method]
    p = np.concatenate([with_links([1e6, 0.1, -1.0, 0.2, 0, 0, 0, 0, 0.67]), sum(REF_TRUTH, []),
                        [0.4, 2.0, NAT_WIDTH, 0.0]])
    ti = np.asarray(m5.TI(REF_X, p, ['Singlet'] * 3, JN, SerialPool(), mp['x0'], mp['MulCo'], mp['INS'], [0], [0],
                          Met=mp['Met'], Norm=norm, smearing=smearing), float)
    assert np.allclose(model.counts(p), ti, rtol=1e-12, atol=0)
    bumped = []
    for i in (0, 2, 3, NB, NB + 1, NB + 3, NB + 9):
        b = p.copy()
        b[i] *= 1 + 1e-6
        bumped.append(with_links(np.append(b, 0.67)))
    for got, b in zip(model.counts_of_many(bumped), bumped):
        assert np.allclose(got, model.counts(b), rtol=1e-12, atol=0)
    m5.FIT_CANCEL.set()
    try:
        with pytest.raises(m5.FitInterrupted):
            model.counts(p)
    finally:
        m5.FIT_CANCEL.clear()


def test_reference_fit_finds_the_lines_and_takes_the_number_set():
    """SMS: Nnr = 0. The growth stops where the Fit quality is good; the
    quality the fitter computes with the counts' noise prepared once is the
    one fit_quality computes from scratch."""
    from syncmoss.fit_quality import fit_quality
    from syncmoss.spectrum_reference import ReferenceFitter, quality_marks
    fitter = ReferenceFitter(REF_X, _reference(REF_TRUTH), _lines_model()[0], 'SMS')
    assert not fitter.edges_absorb
    fit = fitter.automatic()
    lines = fit.lines[np.argsort(fit.lines[:, 1])]
    assert len(lines) == 2 and np.all(fit.p[4:NB] == [0.0, fit.p[1], fit.p[2], fit.p[3]])
    assert np.allclose(lines[:, 1], [-1.0, 0.8], atol=0.02)
    assert np.allclose(lines[:, 0], [1.0, 0.6], rtol=0.1)
    quality = fitter.quality(fit)
    assert quality_marks(quality) == []
    n_free = len(fitter.free_indices(2))
    again = fit_quality([(REF_X, fitter.y, fit.model, fitter.model.counts(fit.p[:NB]), np.ones(len(REF_X), bool))],
                        fit.chi2, fit.deviance / (len(REF_X) - n_free), n_free)[0]
    assert again['rows'] == quality['rows']
    for n in (3, 1):
        assert len(fitter.with_number(fit, n).lines) == n
    # one line less is worse by far, one more gains next to nothing
    assert fitter.with_number(fit, 1).deviance - fit.deviance > 100
    assert fit.deviance - fitter.with_number(fit, 3).deviance < 30


@pytest.mark.parametrize("fraction", [0.55, 0.9])
def test_a_cms_reference_fits_its_nnr_within_the_bounds(fraction):
    """CMS: Nnr/Ns fitted within NNR_FRACTION (Onr, c²nr, linnr following Os,
    c²s, lins). A thick line's saturated depth tells the counts' Nnr/Ns; thin
    lines cannot tell it from their T, so it stays near the start -- and they
    are not taken for vanished for that (their significance is the fraction's
    held)."""
    from syncmoss.spectrum_reference import NNR_FRACTION, ReferenceFitter
    thick = [[15.0, -0.1, NAT_WIDTH, 0.05]]
    fitter = ReferenceFitter(REF_X, _reference(thick, 'CMS', fraction=fraction), _lines_model('CMS')[0], 'CMS')
    fit = fitter.automatic()
    assert len(fit.lines) == 1 and np.all(fit.p[5:NB] == fit.p[1:4])
    assert fit.p[4] == fit.fraction * fit.p[0] and abs(fit.fraction - fraction) < 3 * fit.errors[-1]
    thin = [[0.4, -1.0, NAT_WIDTH, 0.15], [0.2, 0.8, NAT_WIDTH, 0.3]]
    fitter = ReferenceFitter(REF_X, _reference(thin, 'CMS', fraction=fraction), _lines_model('CMS')[0], 'CMS')
    fit = fitter.automatic()
    lines = fit.lines[np.argsort(fit.lines[:, 1])]
    assert len(lines) == 2 and np.allclose(lines[:, 1], [-1.0, 0.8], atol=0.03)
    assert NNR_FRACTION[0] <= fit.fraction <= NNR_FRACTION[1]


def test_reference_fit_tells_an_absorbing_edge():
    from syncmoss.spectrum_reference import ReferenceFitter
    edge = [[1.0, -3.95, NAT_WIDTH, 0.1]]
    assert ReferenceFitter(REF_X, _reference(edge), _lines_model()[0], 'SMS').edges_absorb


# --- the table row --------------------------------------------------------------

def _model_btn(pt, row):
    return pt.row_widgets[row].layout().itemAt(0).widget().layout().itemAt(1).widget()


@pytest.mark.gui
def test_table_row_read_validate_and_mdl_round_trip(physics_app, tmp_path):
    from syncmoss.model_io import (read_model, validate_user_expressions, _save_model_to_file,
                                   load_model_from_path)
    pt = physics_app.params_table
    pt.select_model(1, 'Singlet')
    pt.select_model(2, 'Spectrum')
    pt.select_model(3, 'Distr')
    text = format_spectrum_lines(LINES)
    pt.expression_value_input(2).setText(text)

    model, p, *_links, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = read_model(physics_app)
    assert model == ['Singlet', 'Spectrum', 'Distr']
    assert len(p) == NB + 4 + 4 + 5
    assert Distri == [text, 'X']                      # model order: line set, then the PDF
    assert list(DistriN) == [NB + 4 + 3, NB + 4 + 4 + 4]
    assert validate_user_expressions(physics_app) == []

    path = str(tmp_path / 'spectrum.mdl')
    _save_model_to_file(physics_app, path)
    for row in range(1, len(pt.row_widgets)):
        pt.clear_row_params(row)
    load_model_from_path(physics_app, path)
    model, p, *_rest = read_model(physics_app)
    assert model == ['Singlet', 'Spectrum', 'Distr'] and _rest[3] == [text, 'X']
    # Load model: the file's components added below, each line set with its row, hidden
    load_model_from_path(physics_app, path, insert_row=4)
    model, p, *_rest = read_model(physics_app)
    assert model == ['Singlet', 'Spectrum', 'Distr'] * 2 and _rest[3] == [text, 'X'] * 2
    assert all(pt.row_widgets[row].layout().itemAt(4).widget().isHidden() for row in (2, 5))

    pt.expression_value_input(2).setText('')
    problems = validate_user_expressions(physics_app)
    assert [(prob['kind'], prob['row']) for prob in problems] == [('Spectrum', 2)]


@pytest.mark.gui
def test_distributions_and_references_the_line_set_refuses(physics_app):
    """A Distr/Corr/Recon on a Spectrum takes δ or G, not T, not the line set,
    nothing beyond; neither a link =[i,Y] nor a p[i] may read the line set's
    slot -- Show model and Fit refuse them, saying why."""
    from syncmoss.model_io import validate_user_expressions
    app = physics_app
    pt = app.params_table
    pt.select_model(1, 'Spectrum')
    pt.expression_value_input(1).setText(format_spectrum_lines(LINES))
    pt.select_model(2, 'Distr')
    for par in (1, 2):
        pt._value_input(2, 0).setText(str(par))
        assert pt.get_invalid_distribution_targets() == [] and app.check_user_expressions("Show model")
    for par, what in ((0, 'T'), (3, 'the line set (no number)'), (4, 'beyond the 4 parameters')):
        pt._value_input(2, 0).setText(str(par))
        assert [(hit['row'], hit['what']) for hit in pt.get_invalid_distribution_targets()] == [(2, what)]
        assert not app.check_user_expressions("Show model") and f"par = {par}" in app.log.toPlainText()
    pt.select_model(2, 'Singlet')
    pt._value_input(2, 1).setText(f'=[{NB + 3},1]')
    links = pt.get_links_to_distributions()
    assert [(link['row'], link['reason'], link['column']) for link in links] == [(2, 'placeholder', 'line set')]
    assert not app.check_user_expressions("Fit") and "line set" in app.log.toPlainText()
    pt._value_input(2, 1).setText('0.5')
    pt.select_model(3, 'Expression')
    pt.expression_value_input(3).setText(f'p[{NB + 3}]*2')
    problems = validate_user_expressions(app)
    assert len(problems) == 1 and "line set" in str(problems[0]['error'])


@pytest.mark.gui
def test_reference_window_fits_refits_and_writes_the_row(physics_app, tmp_path, monkeypatch):
    """The menu's "Spectrum" (after Load model, in its own colour): the .dat
    (its header's SMS instrumental function, no question; an SMS reference has
    no Nnr), the window with its disclaimer, the automatic fit, Refit with
    another number of lines, OK writes the row -- T = Σ T_q, the line set in
    its hidden column -- and the app is free."""
    from PySide6.QtWidgets import QApplication, QWidgetAction
    import syncmoss.spectrum_reference_window as srw
    from syncmoss.model_io import read_model
    from syncmoss.spectrum_io import save_spectrum_with_metadata
    app = physics_app
    app.pool = SerialPool()
    app.jn0_input.setText(str(JN))           # read by the window as Fit reads it
    assert not hasattr(app, 'JN0')           # a fresh window: nothing initialized yet
    path = str(tmp_path / 'reference.dat')
    header, method = iio.build_dat_metadata_lines(app)
    assert method == 'SMS'
    save_spectrum_with_metadata(path, REF_X, _reference(REF_TRUTH), header)
    monkeypatch.setattr(srw.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (path, ''))
    monkeypatch.setattr(srw.QMessageBox, 'exec', lambda self: pytest.fail("asked SMS or CMS"))
    pt = app.params_table
    # the menu: after Load model, in the tool's own colour
    entries = [action.defaultWidget() if isinstance(action, QWidgetAction) else action
               for action in _model_btn(pt, 1).menu().actions() if not action.isSeparator()]
    names = [entry.text() for entry in entries]
    assert names.index('Spectrum') == names.index('Load model') + 1
    assert srw.SPECTRUM_TOOL_COLOR in entries[names.index('Spectrum')].styleSheet()
    pt.select_model_by_button('Spectrum', _model_btn(pt, 1))
    window = app.spectrum_reference_window
    assert pt.model_name_at(1) == 'None'                # nothing written before OK
    assert window.disclaimer.text() == srw.DISCLAIMER and not window.disclaimer.isHidden()

    def finish():
        window.thread.wait()
        QApplication.processEvents()
        assert not app.inprogress and not window.thread.isRunning()

    finish()
    assert window.number.value() == 2 and window.ok_button.isEnabled()
    assert "T<sub>ref</sub>" in window.quality.text() and "χ²" in window.quality.text()
    assert "Nnr/Ns" not in window.quality.text() and window.result['fit'].p[4] == 0.0
    window.number.setValue(3)
    window.refit()
    finish()
    fit = window.result['fit']
    assert len(fit.lines) == 3
    window.accept_reference()
    model, p, *_links, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = read_model(app)
    assert model == ['Spectrum']
    assert np.array_equal(parse_spectrum_lines(Distri[0]), fit.lines)
    assert p[NB] == fit.lines[:, 0].sum() and p[NB + 1] == 0.0 and p[NB + 2] == 0.0
    cells = [pt.row_widgets[1].layout().itemAt(col + 1).widget() for col in range(4)]
    assert [cell.isHidden() for cell in cells] == [False, False, False, True]


@pytest.mark.gui
def test_a_dat_without_instrumental_function_asks_sms_or_cms(physics_app, tmp_path, monkeypatch):
    """A .dat with no instrumental function: the user says SMS or CMS and the
    program's own of that method is used (Cancel: nothing); one with a header
    is not asked about."""
    import syncmoss.spectrum_reference_window as srw
    from syncmoss.spectrum_io import save_spectrum_with_metadata
    app = physics_app
    bare = str(tmp_path / 'bare.dat')
    np.savetxt(bare, np.column_stack((REF_X, _reference(REF_TRUTH))), fmt='%.6f')
    asked = []

    def answer(text):
        monkeypatch.setattr(srw.QMessageBox, 'exec', lambda self: asked.append(self.text()) or 0)
        monkeypatch.setattr(srw.QMessageBox, 'clickedButton',
                            lambda self: next((b for b in self.buttons() if b.text() == text), None))

    answer('CMS')
    cms = srw.reference_method(app, bare)
    assert (cms['method'], cms['source'], cms['INS']) == ('CMS', 'internal', iio.get_internal_gcms(app))
    answer('SMS')
    sms = srw.reference_method(app, bare)
    assert (sms['method'], sms['source']) == ('SMS', 'internal')
    answer(None)
    assert srw.reference_method(app, bare) is None
    assert len(asked) == 3 and all("bare.dat" in text and "SMS or CMS" in text for text in asked)
    with_header = str(tmp_path / 'with_header.dat')
    save_spectrum_with_metadata(with_header, REF_X, _reference(REF_TRUTH), iio.build_dat_metadata_lines(app)[0])
    assert srw.reference_method(app, with_header)['source'] == 'file' and len(asked) == 3
