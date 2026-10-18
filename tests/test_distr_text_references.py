"""p[i] in Distr/Corr texts, checked against the brute-force model.

A Distr/Corr text may read any parameter of the model as p[i] -- here the
slots of a Variables row and of Expressions, as it should (never the
distributed parameter itself, whose field the distribution overwrites).
TImod evaluates the text of a component's OUTER distribution at the top level
and the text of an INNER one inside the copies of a multidimensional
distribution, where its own p is the copies' array; p[i] must be the model's
parameter i either way. The result plot's component curves get the texts with
the numbers put in instead (fitting_io.create_subspectra), as their own p is
the component's alone.

The brute-force model is the same spectrum written out by hand: one plain
Sextet per grid point, its amplitude carrying the product of the weights, its
parameters the axis values and the Corr dependencies, every text evaluated on
the model's p. The numbers are chosen to bite: negative centres, a negative
width (only its square enters), a negative slope and a negative Expression,
'**' and a unary minus next to a reference, a reference used twice and one
inside a bracket raised to a power, scientific notation. Every parameter is
written p[<number>], the one spelling the check before a fit lets through
(test_expression_validation); the L/R of every distribution are links to
Expressions (as in tests/data/Distr_distr.mdl), and two more links have
negative factors. p is prepared as a fit prepares it: the Expressions in
order, then the links.

Few points, Num 2-3 (3 where a density's sign is at stake: two points placed
symmetrically about its centre get equal weights whatever its sign). Pure
NumPy/Numba, serial pool.
"""
import copy
import json
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import numco, number_of_baseline_parameters as NB
from syncmoss.expression_eval import eval_expr
from syncmoss.fitting_io import compute_component_curves, create_subspectra
from syncmoss.instrumental_io import source_for

from conftest import FROZEN_PARAMETERS, SerialPool

with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _SEXTET = list(json.load(_f)["cases"]["Sextet"]["params"])
_SINGLET = [1.0, 0.0, 0.098, 0.15]                  # T, delta (a link sets it), L, G
JN = 16
_X = np.linspace(-10.0, 10.0, 41)
_G = float(np.genfromtxt(os.path.join(FROZEN_PARAMETERS, "GCMS.txt"), delimiter='\t'))
_METHOD = dict(x0=0.0, MulCo=MulCoCMS, INS=_G, Met=1)
_BASE = [1.0e4] + [0.0] * (NB - 1)
_SLOTS = {'Singlet': 4, 'Sextet': 14, 'Distr': 5, 'Corr': 2, 'Recon': 7, 'Variables': numco,
          'Expression': 1, 'Nbaseline': NB}

# Sextet slots: T, delta, eps, H, L, G, theta, phi, A, Am, a+, a-, GH, I1/I3
DELTA, EPS, H, G, A_MINUS = 1, 2, 3, 5, 11

# The Variables row: slot -> value
_VARIABLES = {0: -0.04,       # v0: delta centre
              1: -0.03,       # v1: delta width -- negative, only its square enters
              2: 33.0,        # v2: H centre
              3: 1.2,         # v3: H width
              4: -0.5,        # v4: a slope
              5: 2.5e-3,      # v5: a small coefficient
              6: -1.5e-3}     # v6: a small negative one
# The Expressions, in table order (e0 ... e6)
_EXPRESSIONS = ["-sqrt(p[{v3}])/20",              # e0: negative
                "p[{v2}] - 2*p[{v3}]",            # e1, e2: L, R of the H distribution
                "p[{v2}] + 2*p[{v3}]",
                "p[{v0}] - 3*abs(p[{v1}])",       # e3, e4: L, R of the delta one
                "p[{v0}] + 3*abs(p[{v1}])",
                "p[{e0}] - 0.02",                 # e5, e6: L, R of the eps one
                "p[{e0}] + 0.02"]
_RANGE = {DELTA: ('e3', 'e4'), EPS: ('e5', 'e6'), H: ('e1', 'e2')}

_PDF_DELTA = "exp(-(X-p[{v0}])**2/2/p[{v1}]**2)"            # negative centre and width before **
_PDF_H = "exp(-(X-p[{v2}])**2/(2*p[{v3}]*p[{v3}]))"          # a reference used twice
_PDF_H_POWER = "exp(-((X-p[{v2}])/p[{v3}])**2/2)/p[{v3}]"     # inside a bracket raised to a power
_PDF_EPS = "exp(-(X-p[{e0}])**2/2/(4*p[{v5}])**2)"           # centred at a negative Expression
_CORR_EPS = "p[{v4}]*(X-p[{v0}])*0.1 + p[{e0}]**2"            # negative slope, negative Expression squared
_CORR_EPS_H = "-p[{v4}]**2*0.04 + p[{v6}]*(X-p[{v2}])"        # unary minus before a reference
_CORR_G = "0.12 + p[{v5}]*abs(X - p[{v2}])"
_CORR_DELTA = "p[{v0}] + 2.5e-3*(X-p[{v2}])"                  # scientific notation


class Level:
    """A Distr ('D', a density text) or Recon ('R', weights) over Sextet slot
    *par*, with the Corr rows that follow it: [(slot, dependency text), ...]."""

    def __init__(self, kind, par, num_or_weights, density=None, corrs=()):
        self.kind, self.par = kind, par
        self.weights = None if kind == 'D' else np.asarray(num_or_weights, dtype=float)
        self.num = num_or_weights if kind == 'D' else len(num_or_weights)
        self.density = density
        self.corrs = list(corrs)

    def rows(self):
        return [{'D': 'Distr', 'R': 'Recon'}[self.kind]] + ['Corr'] * len(self.corrs)

    def block(self):
        """par, L, R (0: links set them), Num, then D_dif, D_dif2 and the weights
        placeholder of a Recon or the density placeholder of a Distr, then
        (par, placeholder) for every Corr."""
        block = [float(self.par), 0.0, 0.0, float(self.num)]
        block += [0.0] if self.kind == 'D' else [0.0, 0.0, 0.0]
        for par_c, _ in self.corrs:
            block += [float(par_c), 0.0]
        return block


_CASES = {
    "outer Distr + Corr": [Level('D', H, 3, _PDF_H, corrs=[(DELTA, _CORR_DELTA)])],
    "inner + outer": [Level('D', DELTA, 3, _PDF_DELTA), Level('D', H, 3, _PDF_H)],
    "inner and outer with Corr": [Level('D', DELTA, 3, _PDF_DELTA, corrs=[(EPS, _CORR_EPS)]),
                                  Level('D', H, 3, _PDF_H_POWER, corrs=[(G, _CORR_G)])],
    "three levels": [Level('D', DELTA, 3, _PDF_DELTA), Level('D', EPS, 2, _PDF_EPS),
                     Level('D', H, 3, _PDF_H, corrs=[(G, _CORR_G)])],
    "Distr inside a Recon": [Level('D', DELTA, 3, _PDF_DELTA, corrs=[(EPS, _CORR_EPS)]),
                             Level('R', H, [0.2, 0.5, 0.3], corrs=[(G, _CORR_G)])],
    "Recon inside a Distr": [Level('R', DELTA, [0.6, 0.4]),
                             Level('D', H, 3, _PDF_H, corrs=[(EPS, _CORR_EPS_H)])],
}


@pytest.fixture(autouse=True)
def _empty_store():
    m5._NODE_SUMS.clear()
    yield
    m5._NODE_SUMS.clear()


def _flat_start(model, row):
    return NB + sum(_SLOTS[name] for name in model[:row])


def _names(model):
    """The flat index of every Variables slot (v0, ...) and Expression (e0, ...)."""
    v_start = _flat_start(model, model.index('Variables'))
    names = {f'v{k}': v_start + k for k in range(numco)}
    names.update({f'e{k}': v_start + numco + k for k in range(len(_EXPRESSIONS))})
    return names


def build(levels, nbaseline=False):
    """The model -- a Singlet, then the Sextet with the levels, the Variables
    row and the Expressions; with *nbaseline* the Sextet and its levels form a
    second spectrum -- with p prepared as a fit prepares it. A dict: model, p,
    Distri, Cor, Recon, sextet (flat start of the Sextet), blocks (flat start
    of every level), lengths."""
    chain_rows = ['Sextet'] + [row for level in levels for row in level.rows()]
    chain_params = list(_SEXTET) + [x for level in levels for x in level.block()]
    tail_rows = ['Variables'] + ['Expression'] * len(_EXPRESSIONS)
    tail_params = [0.0] * (numco + len(_EXPRESSIONS))
    if nbaseline:
        model = ['Singlet'] + tail_rows + ['Nbaseline'] + chain_rows
        p = _BASE + _SINGLET + tail_params + _BASE + chain_params
    else:
        model = ['Singlet'] + chain_rows + tail_rows
        p = _BASE + _SINGLET + chain_params + tail_params
    sextet = _flat_start(model, model.index('Sextet'))
    names = _names(model)

    p = np.array(p, dtype=float)
    for k, value in _VARIABLES.items():
        p[names[f'v{k}']] = value
    for k, text in enumerate(_EXPRESSIONS):                      # the Expressions in order,
        p[names[f'e{k}']] = eval_expr(text.format(**names), p)
    blocks, start = [], sextet + len(_SEXTET)                     # then the links
    for level in levels:
        blocks.append(start)
        lo, hi = _RANGE[level.par]
        p[start + 1] = p[names[lo]] * 1                          # L = =[e.., 1]
        p[start + 2] = p[names[hi]] * 1                          # R = =[e.., 1]
        start += len(level.block())
    p[sextet + A_MINUS] = p[names['v6']] * -2                    # a- = =[v6, -2]
    p[NB + 1] = p[names['v0']] * -10                              # Singlet delta = =[v0, -10]

    return dict(model=model, p=p, names=names, sextet=sextet, blocks=blocks,
                Distri=[level.density.format(**names) for level in levels if level.kind == 'D'],
                Cor=[text.format(**names) for level in levels for _, text in level.corrs],
                Recon=[level.weights for level in levels if level.kind == 'R'],
                lengths=[len(_X), len(_X)] if nbaseline else None)


def brute_force_copies(levels, case):
    """The Sextet copies written out by hand: (rows, flat params)."""
    p, names = case['p'], case['names']
    base = list(p[case['sextet']:case['sextet'] + len(_SEXTET)])
    copies = [(1.0, {})]
    for level, start in zip(levels, case['blocks']):
        X = np.linspace(p[start + 1], p[start + 2], int(p[start + 3]))
        ge = (eval_expr(level.density.format(**names), p, X) + 0 * X if level.kind == 'D'
              else np.array(level.weights))
        ge = ge / np.sum(ge)
        corr_values = [(par_c, eval_expr(text.format(**names), p, X) + 0 * X)
                       for par_c, text in level.corrs]
        copies = [(weight * ge[i], {**assign, level.par: X[i],
                                    **{par_c: values[i] for par_c, values in corr_values}})
                  for weight, assign in copies for i in range(level.num)]
    params = []
    for weight, assign in copies:
        one = list(base)
        for slot, value in assign.items():
            one[slot] = value
        one[0] = base[0] * weight
        params += one
    return ['Sextet'] * len(copies), params


def _ti(model, p, distri=(), cor=(), recon=(), norm=1, lengths=None):
    x = np.concatenate([_X] * len(lengths)) if lengths else _X
    return np.asarray(m5.TI(x, np.asarray(p, dtype=float), list(model), JN, SerialPool(), 0.0, MulCoCMS,
                            _G, list(distri) or [0], list(cor) or [0], Met=1, Norm=norm,
                            Recon=list(recon) or [0], lengths=lengths), dtype=float)


def _assert_same(got, expected, label):
    depth = np.max(expected) - np.min(expected)
    assert np.max(np.abs(got - expected)) / depth < 1e-9, label


def _singlet(case):
    return list(case['p'][:NB + len(_SINGLET)])                   # baseline + Singlet


def test_every_text_here_is_one_the_check_lets_through():
    from syncmoss.model_io import p_spelling_problem
    for levels in _CASES.values():
        case = build(levels)
        texts = case['Distri'] + case['Cor'] + [e.format(**case['names']) for e in _EXPRESSIONS]
        for text in texts:
            assert p_spelling_problem(text) is None, text


@pytest.mark.parametrize("label", list(_CASES))
def test_the_model_is_the_brute_force_model(label):
    levels = _CASES[label]
    case = build(levels)
    rows, copies = brute_force_copies(levels, case)
    got = _ti(case['model'], case['p'], case['Distri'], case['Cor'], case['Recon'])
    _assert_same(got, _ti(['Singlet'] + rows, _singlet(case) + copies), label)
    # ...and not vacuously: the distribution changes the spectrum
    sextet = list(case['p'][case['sextet']:case['sextet'] + len(_SEXTET)])
    plain = _ti(['Singlet', 'Sextet'], _singlet(case) + sextet)
    assert np.max(np.abs(got - plain)) / (np.max(plain) - np.min(plain)) > 1e-3, label


@pytest.mark.parametrize("label", list(_CASES))
def test_the_component_curve_is_the_brute_force_component(label):
    """The result plot's curve of the distributed Sextet: create_subspectra puts
    the numbers into the texts, the component is computed on its own p."""
    levels = _CASES[label]
    case = build(levels)
    Ps, Psm, Distri_t, Cor_t, _, _ = create_subspectra(case['model'], case['Distri'], case['Cor'], case['p'])
    curves, _ = compute_component_curves(_X, Ps, Psm, Distri_t, Cor_t, JN, SerialPool(), _METHOD,
                                         m5.SMS_POL_DEFAULT, Recon=case['Recon'])
    norm, _ = source_for(SerialPool(), _X, JN, _METHOD)
    rows, copies = brute_force_copies(levels, case)
    expected = _ti(rows, list(case['p'][:NB]) + copies, norm=norm)
    _assert_same(np.asarray(curves[1], dtype=float), expected, label)


@pytest.mark.parametrize("text", [
    "exp(-(X-p[1])**2/2/p[2]**2)",           # a negative value before **
    "-p[3]**2*X + p[0]",                     # a unary minus before a negative one
    "p[2]**p[0]*X",                          # a power of two numbers
    "X**2*p[4] - p[5]*X",                    # tiny and huge, in scientific notation
    "p[6]*X + p[3]",                         # -0.0
])
def test_the_numbers_go_into_a_text_with_their_value(text):
    """create_subspectra's substitution: the text with the numbers put in
    gives what the text gives on p, to the last bit."""
    from syncmoss.fitting_io import _substitute_p_refs
    p = np.array([2.0, -0.04, -0.03, -0.5, 1.0e-20, -3.0e300, -0.0])
    X = np.linspace(-0.13, 0.05, 5)
    substituted = _substitute_p_refs(text, p)
    assert 'p[' not in substituted
    assert np.array_equal(eval_expr(substituted, None, X) + 0 * X, eval_expr(text, p, X) + 0 * X)


def test_a_second_spectrum_reads_the_first_spectrums_parameters():
    """Nbaseline: the texts of the second spectrum's distributions read the
    Variables and Expressions of the first spectrum by their flat index."""
    levels = _CASES["inner and outer with Corr"]
    case = build(levels, nbaseline=True)
    rows, copies = brute_force_copies(levels, case)
    got = _ti(case['model'], case['p'], case['Distri'], case['Cor'], case['Recon'], lengths=case['lengths'])
    expected = _ti(['Singlet', 'Nbaseline'] + rows, _singlet(case) + _BASE + copies, lengths=case['lengths'])
    _assert_same(got, expected, "Nbaseline")


def test_every_recon_weight_list_reads_the_texts_as_ti():
    """TI_recon_variants (a fit's Recon weight columns) computes the copies of
    the inner Distr once and recombines them per weight list: its texts too
    must read the model's p."""
    case = build(_CASES["Distr inside a Recon"])
    weight_lists = [np.array([0.2, 0.5, 0.3]), np.array([0.5, 0.3, 0.2])]
    got = m5.TI_recon_variants(_X, case['p'], case['model'], JN, SerialPool(), 0.0, MulCoCMS, _G,
                               case['Distri'], case['Cor'], 1, 1, m5.SMS_POL_DEFAULT,
                               [[w] for w in weight_lists])
    for weights, spectrum in zip(weight_lists, got):
        levels = copy.deepcopy(_CASES["Distr inside a Recon"])
        levels[1].weights = weights
        rows, copies = brute_force_copies(levels, case)
        _assert_same(np.asarray(spectrum, dtype=float), _ti(['Singlet'] + rows, _singlet(case) + copies),
                     f"weights {weights}")
