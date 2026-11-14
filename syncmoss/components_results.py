"""The results folder of a search for the components of many spectra
(components_search; docs/instead_of_pca.md §6).

Everything the search makes goes there, so that the window opens it later
without searching again:
* run.json -- the settings (instrumental function, known part, the CMS
  non-resonant counts, the spectra and their layout) and the record of the
  search, with every component's lines and strengths;
* result.npz -- the arrays the window draws: the velocities fitted, the counts,
  every spectrum's weights with their errors and covariances, baseline and fit;
* lines.mdl -- every line found, one Singlet row each, T its mean over the
  spectra (the copies of one line in several components as one line):
  openable as a model;
* component_<k>.mdl -- component k as one 'Spectrum' row, its line set, T its
  mean over the spectra where it is significant (w_sk above the 99 % normal
  point times its error): a row for the user's own fits;
* component_<k>.dat -- what Create spectrum from model makes of that row: the
  spectra's velocities, no noise, the instrumental function in the header;
  its baseline the off-resonance counts of the spectra where the component is
  significant, summed, so that the errors of a fit to it show roughly what
  those spectra support;
* components_T.txt, lines_T.txt -- every component's and every line's T in
  every spectrum with its error, the spectrum's coordinates, baseline and fit.

The .mdl rows carry the bounds and fixes a table row of that model starts
with (tests/test_components_search.py compares them with the table's).
"""
import json
import os

import numpy as np

from syncmoss.components_search import Components, normal_point
from syncmoss.constants import NAT_WIDTH, model_colors
from syncmoss.model_io import write_result_model
from syncmoss.spectrum_io import compute_model_spectrum, save_spectrum_with_metadata
from syncmoss.spectrum_model import format_spectrum_lines

RUN_FILE, ARRAYS_FILE = 'run.json', 'result.npz'
LINES_MODEL, COMPONENTS_TABLE, LINES_TABLE = 'lines.mdl', 'components_T.txt', 'lines_T.txt'


def component_file(k, extension):
    return f'component_{k + 1}.{extension}'


# --- what the numbers say ------------------------------------------------------------

def significant(w, w_err):
    """(n, K): where a component's weight is above the 99 % normal point times its error."""
    return np.asarray(w) > normal_point() * np.asarray(w_err)


def component_T(w, w_err):
    """Every component's T: its mean over the spectra where it is
    significant (nan where it is nowhere)."""
    sig = significant(w, w_err)
    count = sig.sum(axis=0)
    return np.where(count > 0, np.where(sig, w, 0.0).sum(axis=0) / np.maximum(count, 1), np.nan)


def unique_lines(res):
    """The lines with the copies of one line (res.cols) as one: (lines (Q, 3)
    -- T their summed mean T, δ and G their T-weighted means --, for every
    column the line it is)."""
    ids = sorted(set(res.cols))
    which = np.array([ids.index(c) for c in res.cols])
    T_col = (res.w @ res.beta).mean(axis=0)
    out = np.zeros((len(ids), 3))
    for q in range(len(ids)):
        cols = np.flatnonzero(which == q)
        share = np.maximum(T_col[cols], 0.0)
        share = share / share.sum() if share.sum() > 0 else np.full(len(cols), 1 / len(cols))
        out[q] = [T_col[cols].sum(), share @ res.lines[cols, 1], share @ res.lines[cols, 2]]
    return out, which


def line_T(res, w_cov):
    """Every spectrum's T of every line (the copies summed) as the components
    give it, T_sq = Σ_k w_sk β_kq, and its error from the weights' covariance
    of each spectrum: (n, Q) each."""
    lines, which = unique_lines(res)
    B = np.zeros((len(res.beta), len(lines)))
    for j, q in enumerate(which):
        B[:, q] += res.beta[:, j]
    T = res.w @ B
    err = np.sqrt(np.maximum(np.einsum('kq,skl,lq->sq', B, w_cov, B), 0.0))
    return T, err


# --- model files -------------------------------------------------------------------

def _field(value, lower='', upper='', fixed=False):
    return [str(value), str(lower), str(upper), '', 'True' if fixed else 'False']


def baseline_fields(ns, middle, nonres_link):
    """The baseline row as the table starts it -- Ns free (at least 1), the
    rest fixed, Onr, c²nr, linnr linked to Os, c²s, lins -- with Ns, Os the
    middle of the velocities and Nnr *nonres_link* (0, or '=[0,r]': r Ns)."""
    return ([_field(repr(float(ns)), 1), _field(repr(float(middle)), fixed=True), _field(0, fixed=True),
             _field(0, fixed=True), _field(nonres_link, fixed=True)]
            + [_field(f'=[{col},1]', fixed=True) for col in (1, 2, 3)])


def singlet_fields(T, delta, G):
    """A Singlet row as the table starts one: T >= 0, the natural L fixed, G >= 0."""
    return [_field(repr(float(T)), 0), _field(repr(float(delta))), _field(repr(NAT_WIDTH), NAT_WIDTH, fixed=True),
            _field(repr(float(G)), 0)]


def spectrum_fields(T, lines):
    """A 'Spectrum' row as the table starts one -- T >= 0, its shift free, its
    G fixed at 0 -- with the line set *lines* (T_q, δ_q, G_q)."""
    return [_field(repr(float(T)), 0), _field(0.0), _field(0.0, 0, fixed=True),
            _field(format_spectrum_lines(lines), fixed=True)]


def write_model(path, rows):
    """A .mdl of the baseline and the component rows *rows* = [(model, fields)]."""
    names = [name for name, _ in rows]
    colors = [model_colors[k % len(model_colors)] for k in range(len(rows))]
    write_result_model(path, (names, colors, [fields for _, fields in rows]))


# --- the folder ----------------------------------------------------------------------

def write_results(folder, res, job, pool):
    """Every file of the module docstring into *folder*. *job* is what the
    search was given (the window prepares it): 'x' the velocities fitted,
    'Y' their counts, 'A' every velocity of the spectra, 'files',
    'coordinates' ((n, 2) or None for a series), 'method_params' (resolved,
    with the drive of A), 'header' (the instrumental lines of a .dat), 'JN',
    'pol', 'known' ([(row, parameters)]), 'nonres' ((r, C_high, C_static)),
    'settings' (what run.json says of them). Returns the paths written."""
    os.makedirs(folder, exist_ok=True)
    x, Y, A = (np.asarray(job[key], dtype=float) for key in ('x', 'Y', 'A'))
    n, K = res.w.shape
    w_cov = res.w_cov
    middle = 0.5 * (A.min() + A.max())
    ratio, c_high, c_static = job['nonres']
    nonres_link = 0 if ratio == 0 else f'=[0,{ratio!r}]'
    written = []

    def path(name):
        written.append(os.path.join(folder, name))
        return written[-1]

    # lines.mdl: every line one Singlet
    lines, which = unique_lines(res)
    level = float(np.mean(res.base[:, 0]))
    write_model(path(LINES_MODEL), [('baseline', baseline_fields(level, middle, nonres_link))]
                + [('Singlet', singlet_fields(*line)) for line in lines])
    # component_<k>: its 'Spectrum' row, and the spectrum Create spectrum from model makes of it
    T_k = component_T(res.w, res.w_err)
    sig = significant(res.w, res.w_err)
    for k in range(K):
        own = np.flatnonzero(res.mask[k])
        line_set = np.column_stack([res.beta[k, own], res.lines[own, 1], res.lines[own, 2]])
        T = float(T_k[k]) if np.isfinite(T_k[k]) else float(res.w[:, k].mean())
        where = sig[:, k] if sig[:, k].any() else np.ones(n, dtype=bool)
        ns = float(res.base[where, 0].sum())
        write_model(path(component_file(k, 'mdl')),
                    [('baseline', baseline_fields(ns, middle, nonres_link)), ('Spectrum', spectrum_fields(T, line_set))])
        p = np.array([ns, middle, 0, 0, ratio * ns + where.sum() * c_high, middle, 0, 0, T, 0, 0, 0], dtype=float)
        counts = compute_model_spectrum(dict(A=A, p=p, model=['Spectrum'], Distri=[format_spectrum_lines(line_set)],
                                             Cor=[0], Recon=[0], JN=job['JN'], pol=job['pol'],
                                             method_params=job['method_params']), pool) + where.sum() * c_static
        save_spectrum_with_metadata(
            path(component_file(k, 'dat')), A, counts,
            list(job['header']) + [f"# Component {k + 1} of {len(job['files'])} spectra, T {T!r}, without noise "
                                   f"and without the known part; baseline: the {int(where.sum())} spectra "
                                   f"where it is significant, summed (SYNCmoss)"])
    # the tables
    coords = job.get('coordinates')
    head = ['spectrum', 'file'] + (['X', 'Y'] if coords is not None else [])

    def lead(s):
        return [str(s + 1), os.path.basename(job['files'][s])] + \
            ([repr(float(v)) for v in coords[s]] if coords is not None else [])
    with open(path(COMPONENTS_TABLE), 'w', encoding='utf-8') as f:
        f.write('\t'.join(head + sum(([f'T{k + 1}', f'dT{k + 1}'] for k in range(K)), [])
                          + ['Ns', 'lins', 'c2s', 'deviance', 'D_norm', 'D_norm_1sigma']) + '\n')
        for s in range(n):
            values = sum(([res.w[s, k], res.w_err[s, k]] for k in range(K)), []) + list(res.base[s]) + \
                [res.deviance[s], res.dnorm[s], res.dnorm_sigma[s]]
            f.write('\t'.join(lead(s) + [repr(float(v)) for v in values]) + '\n')
    T, err = line_T(res, w_cov)
    with open(path(LINES_TABLE), 'w', encoding='utf-8') as f:
        f.write('\t'.join(head + sum(([f'T({d:+.3f})', f'dT({d:+.3f})'] for _, d, _ in lines), [])) + '\n')
        for s in range(n):
            f.write('\t'.join(lead(s) + [repr(float(v)) for q in range(len(lines)) for v in (T[s, q], err[s, q])])
                    + '\n')
    np.savez(path(ARRAYS_FILE), x=x, A=A, Y=Y, w=res.w, w_err=res.w_err, w_cov=w_cov, base=res.base,
             deviance=res.deviance, dnorm=res.dnorm, dnorm_sigma=res.dnorm_sigma, lines=res.lines, beta=res.beta,
             mask=res.mask, cols=np.asarray(res.cols), coordinates=np.zeros((0, 2)) if coords is None else coords)
    components = [dict(T=float(T_k[k]), lines=[[float(res.beta[k, j]), float(res.lines[j, 1]), float(res.lines[j, 2])]
                                              for j in np.flatnonzero(res.mask[k])]) for k in range(K)]
    with open(path(RUN_FILE), 'w', encoding='utf-8') as f:
        json.dump(dict(settings=job['settings'], files=list(job['files']), components=components,
                       record=res.record), f, indent=1, default=_plain)
    return written


def _plain(value):
    """What json cannot write: numbers and arrays of numpy."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not written")


def read_results(folder):
    """(run.json's dict, the components found (components_search.Components),
    result.npz's arrays) of a results folder."""
    with open(os.path.join(folder, RUN_FILE), encoding='utf-8') as f:
        run = json.load(f)
    with np.load(os.path.join(folder, ARRAYS_FILE)) as data:
        a = {key: data[key] for key in data.files}
    res = Components(a['lines'], a['beta'], a['mask'], [int(c) for c in a['cols']], a['w'], a['w_cov'], a['base'],
                     a['deviance'], a['dnorm'], a['dnorm_sigma'], run['record'])
    return run, res, a
