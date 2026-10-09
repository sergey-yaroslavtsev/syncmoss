"""Simultaneous one-model fit: the model of the table fitted to all the spectra
of the path box at once.

The model is built for ONE spectrum (the "template"). Every free parameter is
shared by all the spectra -- one value for all of them -- except the ones marked
independent with ``=(X)``: those get a value of their own in every spectrum, all
starting from X. Every spectrum has its own baseline. N, N1, N2, ... in the
formulas take each spectrum's own values (spectrum_parameters).

How: the template is expanded into the form of a hand-built Nbaseline model --
one section per spectrum, ``template + [Nbaseline + template] * (K - 1)`` -- and
fitted by the ordinary simultaneous fit (fitting_io.fit_model). Every section is
as long as the template (L slots: an Nbaseline has the baseline's 8) and the
first one sits at the template's own indices, so spectrum k's copy of template
slot i is slot ``k*L + i``:

* the copies of a shared parameter in sections 2..K are links to section 1's
  slot;
* a reference to template slot i -- the source of a link ``=[i,Y]``, a ``p[i]``
  in an Expression/Distr/Corr text -- points at section 1 when i is shared, and
  otherwise at the section's own copy (a baseline, independent, fixed, linked or
  Expression parameter). So a link never follows another link (read_model has
  already followed the template's chains to their end), as minimi_hi applies
  the links in one pass;
* the baseline of spectra 2..K starts from the spectrum's own counts (Ns from
  spectrum_io.calculate_backgrounds, as in a sequence) unless Ns is a link; its
  other values are the template's, its links stay inside the spectrum;
* a Recon's weights are shared: every copy uses section 1's weights
  (recon_fit_layout's *share*).

Section 1 of the result is therefore the template's own layout: spectrum 1's part
of the result is ``p[:L]``, and Take result writes nothing else.
"""
import os
import re

import numpy as np

from syncmoss import fitting_io
from syncmoss.constants import number_of_baseline_parameters as NB, NBASELINE_COLOR
from syncmoss.model_io import (read_model, read_bounds_and_fix, model_file_rows,
                               fitted_model_rows)
from syncmoss.spectrum_io import calculate_backgrounds
from syncmoss.spectrum_parameters import SpectrumParameters, substitute

_P_REF = re.compile(r'p\[(\d+)\]')


def read_template(app):
    """The model of the parameters table, as the one-model fit starts from it.

    Read in the GUI thread when the fit (or Show model) starts: the formulas as
    typed (with N, N1, ...), the start values, bounds and fix boxes, which
    parameters are independent -- and what the results table and Take result
    need of it: the colours and parameter names of the components, the texts as
    typed, the links and independent values, the model rows.
    """
    model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN, Recon, ReconN = \
        read_model(app, substitute_names=False)
    p = np.array(p, dtype=float)
    bounds, fix = read_bounds_and_fix(app, len(p))
    pt = app.params_table
    return {
        'model': list(model), 'p': p,
        'con1': np.array(con1, dtype=float), 'con2': np.array(con2, dtype=float),
        'con3': np.array(con3, dtype=float),
        'Distri': list(Distri), 'Cor': list(Cor), 'Expr': list(Expr),
        'NExpr': np.array(NExpr, dtype=int), 'DistriN': np.array(DistriN, dtype=float),
        'Recon': [np.array(w, dtype=float) for w in Recon],
        'ReconN': np.array(ReconN, dtype=float),
        'bounds': np.array(bounds, dtype=float), 'fix': np.array(fix, dtype=int),
        'independent': list(pt.get_independent_slots()),
        'colors': pt.get_component_colors(),
        'names': pt.get_parameter_names(),
        'texts': pt.get_expression_texts(),
        'links': pt.get_link_snapshot(),
        'rows': model_file_rows(app),
    }


def shared_slots(template):
    """Template slots fitted with ONE value for all the spectra: every free
    parameter outside the baseline that is not independent, a link or a
    placeholder (Expression, Distr, Recon)."""
    fixed = {int(i) for i in np.ravel(template['fix'])}
    special = ({int(i) for i in template['con1']} | {int(i) for i in template['NExpr']}
               | {int(i) for i in template['DistriN']} | {int(i) for i in template['ReconN']})
    independent = {int(i) for i in template['independent']}
    return [i for i in range(NB, len(template['p']))
            if i not in fixed and i not in special and i not in independent]


def expand(template, spectra, backgrounds=None):
    """The template expanded over *spectra* (SpectrumParameters, in order).

    Returns the inputs of fitting_io.fit_model (model, p, links, texts, slots,
    Recon, bounds, fix, recon_share) plus 'L' (the template's length) and
    'shared' (shared_slots). *backgrounds* holds each spectrum's estimated
    counts; spectrum 1 keeps the template's Ns.
    """
    p = np.asarray(template['p'], dtype=float)
    L = len(p)
    shared = shared_slots(template)
    is_shared = set(shared)
    linked = {int(i) for i in template['con1']}
    fixed = sorted({int(i) for i in np.ravel(template['fix'])})
    recons = len(template['Recon'])

    def where(i, k):
        """Section k's copy of template slot i -- section 1's when i is shared."""
        return i if i in is_shared else k * L + i

    def text_for(text, k):
        renumbered = _P_REF.sub(lambda m: f"p[{where(int(m.group(1)), k)}]", str(text))
        return substitute(renumbered, spectra[k])

    model, values, bounds, fix = [], [], [[], []], []
    con1, con2, con3 = [], [], []
    Distri, Cor, Expr = [], [], []
    NExpr, DistriN, ReconN = [], [], []
    Recon, recon_share = [], []
    for k in range(len(spectra)):
        offset = k * L
        if k > 0:
            model.append('Nbaseline')
        model.extend(template['model'])

        section = p.copy()
        if k > 0 and backgrounds is not None and 0 not in linked:
            section[0] = float(backgrounds[k])
        values.append(section)
        bounds[0].extend(template['bounds'][0])
        bounds[1].extend(template['bounds'][1])
        fix.extend(offset + i for i in fixed)

        # The template's links, then (sections 2..K) the shared parameters
        # following section 1
        for target, source, factor in zip(template['con1'], template['con2'], template['con3']):
            con1.append(offset + int(target))
            con2.append(where(int(source), k))
            con3.append(float(factor))
        if k > 0:
            for i in shared:
                con1.append(offset + i)
                con2.append(i)
                con3.append(1.0)

        Expr.extend(text_for(text, k) for text in template['Expr'])
        Distri.extend(text_for(text, k) for text in template['Distri'])
        Cor.extend(text_for(text, k) for text in template['Cor'])
        NExpr.extend(offset + int(i) for i in template['NExpr'])
        DistriN.extend(offset + int(i) for i in template['DistriN'])
        ReconN.extend(offset + int(i) for i in template['ReconN'])
        Recon.extend(np.array(w, dtype=float) for w in template['Recon'])
        recon_share.extend(range(recons))

    return {
        'model': model, 'p': np.concatenate(values),
        'con1': np.array(con1, dtype=float), 'con2': np.array(con2, dtype=float),
        'con3': np.array(con3, dtype=float),
        'Distri': Distri, 'Cor': Cor, 'Expr': Expr,
        'NExpr': np.array(NExpr, dtype=int), 'DistriN': np.array(DistriN, dtype=float),
        'Recon': Recon, 'ReconN': np.array(ReconN, dtype=float),
        'bounds': np.array(bounds, dtype=float), 'fix': np.array(fix, dtype=int),
        'recon_share': recon_share, 'L': L, 'shared': shared,
    }


def fit(app, template, spectra, pool):
    """Fit the template to *spectra* (SpectrumParameters) simultaneously.

    Runs in the fitting thread. Returns fitting_io.fit_model's result; when it
    succeeded, its 'one_model' entry is the OneModelResult. Raises
    FitInterrupted when stopped (Interrupt).
    """
    paths = [spectrum.path for spectrum in spectra]
    backgrounds = calculate_backgrounds(paths, app.calibration_path)
    inputs = expand(template, spectra, backgrounds)
    inputs['exclusion_regions'] = app.exclusion_regions_in_use()
    result = fitting_io.fit_model(app, inputs, paths, pool)
    if result.get('success'):
        result['one_model'] = OneModelResult(template, spectra, inputs, result)
    return result


def spectra_of(entries):
    """SpectrumParameters of the path box's (path, values) *entries*: N is the
    position, counted from 1."""
    return [SpectrumParameters(i + 1, values, path) for i, (path, values) in enumerate(entries)]


def parameters_line(spectrum):
    """'N = 3   N1 = 77   N2 = 0' -- a spectrum's own numbers."""
    items = [f"N = {spectrum.number}"] + [
        f"N{i + 1} = {value:g}" for i, value in enumerate(spectrum.values)]
    return "   ".join(items)


class OneModelResult:
    """A finished simultaneous one-model fit, and spectrum by spectrum the
    template's view of it -- what the results table, the result window, Take
    result and Save result show.

    For the result window it is a series of spectra like result_window's
    SequenceSeries and NbaselineSeries (count, label, curves, values, ... per
    spectrum). It shows no correlation matrix there: one spectrum's block of
    the joint covariance means nothing on its own.
    """

    shows_correlations = False

    def __init__(self, template, spectra, inputs, result):
        self.template = template
        self.spectra = list(spectra)
        self.inputs = inputs
        self.result = result
        self.L = inputs['L']
        self.shared = set(inputs['shared'])
        self.p = np.asarray(result['parameters'], dtype=float)
        errors = np.asarray(result['errors'], dtype=float).ravel()
        # minimi_hi gives a single 0 when the fit could not move at all
        self.errors = errors if len(errors) == len(self.p) else np.full(len(self.p), np.nan)
        self.covariance = np.atleast_2d(np.asarray(result['covariance_matrix'], dtype=float))
        free = np.flatnonzero(~np.isnan(self.errors))
        self._covariance_row = {int(index): row for row, index in enumerate(free)}
        self.chi2 = result['chi2']
        self.chi2_spread = result.get('chi2_spread')

    # --- the spectra ----------------------------------------------------------

    def count(self):
        return len(self.spectra)

    def displayed(self):
        """The spectra the main window shows: the first and the last."""
        return [0] if self.count() == 1 else [0, self.count() - 1]

    def label(self, k):
        """'3 of 12 · Fe_77K.dat'."""
        return f"{k + 1} of {self.count()} · {os.path.basename(self.spectra[k].path)}"

    def parameters_line(self, k):
        """'N = 3   N1 = 77   N2 = 0' -- the spectrum's own numbers."""
        return parameters_line(self.spectra[k])

    def title(self):
        return f"simultaneous one-model fit of {self.count()} spectra"

    def path_of(self, k):
        return self.spectra[k].path

    def spectrum_parameters_of(self, k):
        return self.spectra[k]

    def chi2_of(self, k):
        """The fit's one chi2: the spectra were fitted together."""
        return self.chi2

    def chi2_spread_of(self, k):
        return self.chi2_spread

    def model_of(self, k):
        return list(self.template['model'])

    def model_list_of(self, k):
        return self.model_list()

    def colors_of(self, k):
        return list(self.template['colors'])

    def names_of(self, k):
        return [list(names) for names in self.template['names']]

    # --- spectrum k in the template's layout ----------------------------------

    def _index(self, i, k):
        return i if i in self.shared else k * self.L + i

    def values(self, k):
        """Spectrum k's parameters, in the template's layout."""
        return self.p[k * self.L:(k + 1) * self.L].copy()

    def errors_of(self, k):
        """Their errors: a shared parameter's is section 1's (its copies are links)."""
        return np.array([self.errors[self._index(i, k)] for i in range(self.L)])

    def covariance_of(self, k):
        """The covariance of spectrum k's free parameters, in the template's layout."""
        errors = self.errors_of(k)
        rows = [self._covariance_row.get(self._index(i, k)) for i in range(self.L)
                if not np.isnan(errors[i])]
        if not rows or any(r is None or r >= self.covariance.shape[0] for r in rows):
            return np.zeros((len(rows), len(rows)))
        return self.covariance[np.ix_(rows, rows)]

    def fix_of(self, k):
        return np.flatnonzero(np.isnan(self.errors_of(k)))

    def _section_slice(self, items, name, k):
        per_section = self.template['model'].count(name)
        return list(items[k * per_section:(k + 1) * per_section])

    def distri(self, k):
        return self._section_slice(self.result.get('Distri_substituted', []), 'Distr', k)

    def cor(self, k):
        return self._section_slice(self.result.get('Cor_substituted', []), 'Corr', k)

    def recon(self, k):
        return self._section_slice(self.result.get('Recon', []), 'Recon', k)

    def curves(self, k):
        """(A, B, fit, components, line positions, x4 integration check) of spectrum k."""
        r = self.result
        hires = r.get('hires_diff_list') or [None] * self.count()
        return (r['A_list'][k], r['B_list'][k], r['SPC_f_list'][k], r['FS_list'][k],
                r['FS_pos_list'][k], hires[k])

    def exclusion_regions_of(self, k):
        """The exclusion regions the fit left out (the same for every spectrum)."""
        return tuple(self.result.get('exclusion_regions') or ())

    # --- the template, for the tables ------------------------------------------

    def model_list(self):
        """The template's components, baseline first (as the results table lists them)."""
        return ['baseline'] + list(self.template['model'])

    def typed_texts(self):
        """The template's Expression/Distr/Corr texts as typed, a Recon cell
        with the fitted (shared) weights: what Take result writes back."""
        texts = dict(self.template['texts'])
        weights = iter(self.recon(0))
        for component, name in enumerate(self.model_list()):
            if name == 'Recon':
                w = next(weights, None)
                if w is not None:
                    texts[component] = ','.join(f'{v:.6g}' for v in np.ravel(w))
        return texts

    def texts_of(self, k):
        """Spectrum k's texts as the whole (expanded) model has them -- the p[i]
        of its own copies, its N values -- keyed in the template's layout."""
        first = k * (len(self.template['model']) + 1)
        end = first + len(self.template['model']) + 1
        return {component - first: text for component, text in self.expanded_texts().items()
                if first <= component < end}

    def whole_fit(self):
        """The texts use the whole model's p[i]: they are evaluated with all of it."""
        return self.p, self.errors, self.covariance

    def template_view(self):
        """What "Take result" writes into the parameters table: the template with
        spectrum 1's values. Its links and independent values come from the
        snapshot taken when the fit started -- an independent X stays the start
        value it was -- and so do its bounds and expressions."""
        return {
            'model_list': self.model_list(),
            'colors': list(self.template['colors']),
            'names': [list(names) for names in self.template['names']],
            'parameters': self.values(0),
            'errors': self.errors_of(0),
            'texts': self.typed_texts(),
            'links': dict(self.template['links']),
            'rows': self.model_rows(),
            'fix': np.array(self.template['fix'], dtype=int),
        }

    def model_rows(self):
        """The template's rows with spectrum 1's fitted values written in: the
        <base>_result_model.mdl of this fit, ready for another one-model fit."""
        return fitted_model_rows(self.template['rows'], self.values(0), self.recon(0),
                                 keep_independent=True)

    # --- the whole expanded model, for the main results table -----------------

    def expanded_model_list(self):
        return ['baseline'] + list(self.result['model'])

    def expanded_colors(self):
        colors = list(self.template['colors'])
        out = list(colors)
        for _ in range(1, self.count()):
            out += [NBASELINE_COLOR] + colors[1:]
        return out

    def expanded_names(self):
        names = [list(n) for n in self.template['names']]
        out = list(names)
        for _ in range(1, self.count()):
            out += [list(names[0])] + names[1:]
        return out

    def expanded_texts(self):
        """Expression/Distr/Corr texts of every section (renumbered, with that
        spectrum's N values) and the Recon weights, keyed by component index."""
        texts = {}
        sources = {'Expression': iter(self.inputs['Expr']),
                   'Distr': iter(self.inputs['Distri']),
                   'Corr': iter(self.inputs['Cor'])}
        weights = iter(self.result.get('Recon', []))
        for component, name in enumerate(self.expanded_model_list()):
            if name in sources:
                texts[component] = next(sources[name], '')
            elif name == 'Recon':
                w = next(weights, None)
                if w is not None:
                    texts[component] = ','.join(f'{v:.6g}' for v in np.ravel(w))
        return texts

    def section_of_component(self, component):
        """(spectrum k, position in its section) of a component of the expanded
        model list: position 0 is the baseline (Nbaseline), 1.. the template's
        components."""
        per_section = len(self.template['model']) + 1
        return divmod(int(component), per_section)

    # --- the two spectra the main window shows --------------------------------

    def displayed_model(self):
        model = list(self.template['model'])
        out = list(model)
        for _ in self.displayed()[1:]:
            out += ['Nbaseline'] + model
        return out

    def displayed_colors(self):
        colors = list(self.template['colors'])
        out = list(colors)
        for _ in self.displayed()[1:]:
            out += [NBASELINE_COLOR] + colors[1:]
        return out

    def displayed_names(self):
        names = [list(n) for n in self.template['names']]
        out = list(names)
        for _ in self.displayed()[1:]:
            out += [list(names[0])] + names[1:]
        return out

    def displayed_parameters(self):
        return np.concatenate([self.values(k) for k in self.displayed()])
