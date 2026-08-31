"""Every Distr/Corr/Recon combination must equal the model written out BY HAND.

A marker chain is a shorthand for many submodels: it replicates its base
component Num times, giving copy i the amplitude ``ge[i]*T`` and setting its
parameter 'par' to the axis point ``X[i]`` -- plus, per trailing Corr, another
parameter to ``f(X[i])`` (models.TImod: ``pN[0] = ge*p[V-Prev]``,
``pN[int(p[V])] = X``, ``pN[int(p[V])] = eval(Cor)``). Nested levels take the
outer product, and every copy joins the SAME layer.

So the chained model and the explicitly expanded many-submodel model must give
the same spectrum. test_distr_composition / test_recon_composition pin the
Num=1-pinned-to-base identity (each ordering collapses back onto the plain
component); these go the other way and check a GENUINE multi-point distribution
against its hand-written expansion, which is the only thing that catches a wrong
weight, a swapped Corr target or a mis-sliced Distri/Cor/Recon list.

Num is kept at 2-3 per level throughout -- the indexing is what is under test,
not the grid density.
"""
import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.constants import number_of_baseline_parameters

_E = np.linspace(-10.0, 10.0, 96)

# Same fixtures as test_distr_composition: two distinct thick, anisotropic
# sextets, plus a thin singlet for the scalar (multiplicative) path.
_SX = [20.0, 0.0, 0.0, 33.0, 0.098, 0.15, 50.0, 30.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0]
_SY = [15.0, 0.3, 0.1, 28.0, 0.098, 0.15, 40.0, 70.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0]
_SG = [10.0, 0.1, 0.098, 0.12]                       # Singlet: T, delta, L, G


def _c(model, params, distri=None, cor=None, recon=None, mett=0):
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array(model), _E, 0.0, 1.0,
                 np.array([]), list(distri or []), list(cor or []),
                 Met=-1, Mett=mett, Recon=list(recon or [])),
        dtype=float,
    )


def _eval(formula, X):
    """A Distr PDF / Corr dependency on the axis, exactly as TImod evaluates it."""
    return eval(formula, vars(m5), {'X': X}) + 0 * X


class Level:
    """One Distr/Recon marker together with the Corr markers trailing it."""

    def __init__(self, kind, par, lo, hi, num, density, corrs=()):
        self.kind = kind                     # 'D' -> PDF string, 'R' -> weight vector
        self.par, self.lo, self.hi, self.num = par, lo, hi, num
        self.density = density
        self.corrs = list(corrs)             # [(par_index, formula), ...]

    @property
    def axis(self):
        return np.linspace(self.lo, self.hi, self.num)

    @property
    def weights(self):
        ge = (_eval(self.density, self.axis) if self.kind == 'D'
              else np.asarray(self.density, dtype=float).flatten())
        return ge / np.sum(ge, axis=0)

    @property
    def markers(self):
        return [{'D': 'Distr', 'R': 'Recon'}[self.kind]] + ['Corr'] * len(self.corrs)

    @property
    def block(self):
        """The flat slots of this level: par, L, R, Num, (D_dif, D_dif2,) filler,
        then one (par, filler) pair per Corr."""
        block = [float(self.par), self.lo, self.hi, float(self.num)]
        block += [0.0] if self.kind == 'D' else [0.0, 0.0, 0.0]
        for par_c, _ in self.corrs:
            block += [float(par_c), 0.0]
        return block


def chained(base_params, levels):
    """(markers, params, Distri, Cor, Recon) for the chained form."""
    markers, params, distri, cor, recon = [], list(base_params), [], [], []
    for level in levels:
        markers += level.markers
        params += level.block
        if level.kind == 'D':
            distri.append(level.density)
        else:
            recon.append(np.asarray(level.density, dtype=float))
        cor += [formula for _, formula in level.corrs]
    return markers, params, distri, cor, recon


def expanded(base_params, levels):
    """The same thing written out by hand: (params, n_copies)."""
    copies = [(1.0, {})]
    for level in levels:
        X, ge = level.axis, level.weights
        corr_values = [(par_c, _eval(formula, X)) for par_c, formula in level.corrs]
        copies = [
            (weight * ge[i], {**assign, level.par: X[i],
                              **{par_c: values[i] for par_c, values in corr_values}})
            for weight, assign in copies
            for i in range(level.num)
        ]

    params = []
    for weight, assign in copies:
        one = list(base_params)
        for idx, value in assign.items():
            one[idx] = value
        one[0] = base_params[0] * weight             # amplitude carries the density
        params += one
    return params, len(copies)


# (label, base model, base params, levels, models before the base, their params)
_COMBINATIONS = [
    ("D", "Sextet", _SX, [Level('D', 3, 31.0, 35.0, 3, "exp(-X**2/600)")], [], []),
    ("R", "Sextet", _SX, [Level('R', 3, 31.0, 35.0, 3, [0.2, 0.5, 0.3])], [], []),
    ("DC", "Sextet", _SX,
     [Level('D', 3, 31.0, 35.0, 3, "1+0*X", corrs=[(1, "0.01*X")])], [], []),
    ("RC", "Sextet", _SX,
     [Level('R', 3, 31.0, 35.0, 3, [0.2, 0.5, 0.3], corrs=[(1, "0.01*X")])], [], []),
    ("DCC", "Sextet", _SX,
     [Level('D', 3, 31.0, 35.0, 3, "1+X*0.01",
            corrs=[(1, "0.01*X"), (2, "0.002*X")])], [], []),
    ("DD", "Sextet", _SX,
     [Level('D', 1, -0.1, 0.1, 2, "1+0*X"),
      Level('D', 3, 31.0, 35.0, 3, "exp(-X**2/600)")], [], []),
    ("RR", "Sextet", _SX,
     [Level('R', 1, -0.1, 0.1, 2, [0.4, 0.6]),
      Level('R', 3, 31.0, 35.0, 3, [0.2, 0.5, 0.3])], [], []),
    ("DR", "Sextet", _SX,
     [Level('D', 1, -0.1, 0.1, 2, "1+0*X"),
      Level('R', 3, 31.0, 35.0, 3, [0.2, 0.5, 0.3])], [], []),
    ("RD", "Sextet", _SX,
     [Level('R', 1, -0.1, 0.1, 2, [0.4, 0.6]),
      Level('D', 3, 31.0, 35.0, 3, "exp(-X**2/600)")], [], []),
    ("DCD", "Sextet", _SX,
     [Level('D', 1, -0.1, 0.1, 2, "1+0*X", corrs=[(2, "0.5*X")]),
      Level('D', 3, 31.0, 35.0, 3, "exp(-X**2/600)")], [], []),
    ("DDC", "Sextet", _SX,
     [Level('D', 1, -0.1, 0.1, 2, "1+0*X"),
      Level('D', 3, 31.0, 35.0, 3, "exp(-X**2/600)", corrs=[(5, "0.15+0.001*X")])], [], []),
    ("DCDC", "Sextet", _SX,
     [Level('D', 1, -0.1, 0.1, 2, "1+0*X", corrs=[(2, "0.5*X")]),
      Level('D', 3, 31.0, 35.0, 3, "exp(-X**2/600)", corrs=[(5, "0.15+0.001*X")])], [], []),
    ("RCR", "Sextet", _SX,
     [Level('R', 1, -0.1, 0.1, 2, [0.4, 0.6], corrs=[(2, "0.5*X")]),
      Level('R', 3, 31.0, 35.0, 3, [0.2, 0.5, 0.3])], [], []),
    ("DRD", "Sextet", _SX,
     [Level('D', 1, -0.1, 0.1, 2, "1+0*X"),
      Level('R', 2, -0.05, 0.05, 2, [0.3, 0.7]),
      Level('D', 3, 32.0, 34.0, 2, "1+0.1*X")], [], []),
    # scalar (thin) component: multiplies into the spectrum, never touches Smat
    ("D-scalar", "Singlet", _SG,
     [Level('D', 1, -0.2, 0.2, 3, "exp(-X**2/0.02)")], [], []),
    ("DC-scalar", "Singlet", _SG,
     [Level('D', 1, -0.2, 0.2, 3, "1+0*X", corrs=[(3, "0.12+0.05*X")])], [], []),
    # sharing a layer with another thick component (the copies must sum into the
    # SAME Sigma as the neighbour), and separated from it by a Layer marker
    ("D-in-shared-layer", "Sextet", _SY,
     [Level('D', 3, 26.0, 30.0, 3, "1+0.05*X")], ["Sextet"], _SX),
    ("DC-after-Layer", "Sextet", _SY,
     [Level('D', 3, 26.0, 30.0, 3, "1+0.05*X", corrs=[(1, "0.3+0.01*X")])],
     ["Sextet", "Layer"], _SX),
]


@pytest.mark.parametrize("label,base_model,base_params,levels,prefix,prefix_params",
                         _COMBINATIONS, ids=[c[0] for c in _COMBINATIONS])
@pytest.mark.parametrize("mett", [0, 1], ids=["SMS", "CMS"])
def test_chain_equals_hand_written_expansion(label, base_model, base_params, levels,
                                             prefix, prefix_params, mett):
    markers, chain_params, distri, cor, recon = chained(base_params, levels)
    exp_params, n_copies = expanded(base_params, levels)

    y_chain = _c(list(prefix) + [base_model] + markers,
                 list(prefix_params) + chain_params,
                 distri=distri, cor=cor, recon=recon, mett=mett)
    y_expand = _c(list(prefix) + [base_model] * n_copies,
                  list(prefix_params) + exp_params, mett=mett)
    y_base = _c(list(prefix) + [base_model],
                list(prefix_params) + list(base_params), mett=mett)

    depth = np.max(np.abs(1.0 - y_expand))
    assert np.max(np.abs(y_chain - y_expand)) / depth < 1e-9, (
        f"{label} (mett={mett}): chained model differs from its hand-written "
        f"{n_copies}-submodel expansion"
    )
    # ...and the comparison is not vacuous: the distribution really changed the
    # spectrum relative to the undistributed component.
    assert np.max(np.abs(y_chain - y_base)) / depth > 1e-2, (
        f"{label} (mett={mett}): distribution had no visible effect -- test is vacuous"
    )


# --- the same check driven THROUGH the parameters table ---------------------

_GUI_PDF, _GUI_LO, _GUI_HI, _GUI_NUM = "exp(-X**2/600)", 31.0, 35.0, 3
_GUI_CORR = "0.01*X"
_GUI_RLO, _GUI_RHI, _GUI_RW = -0.05, 0.05, [0.3, 0.7]


def _table_spectrum(window, mett):
    """Read the table with read_model and evaluate it (baseline dropped: Met=-1
    starts the flat array at the first component)."""
    from syncmoss.model_io import read_model
    model, p, _c1, _c2, _c3, Distri, Cor, _E_, _NE, _DN, Recon, _RN = read_model(window)
    return np.asarray(
        m5.TImod(_E, np.asarray(p, float)[number_of_baseline_parameters:],
                 np.array(model), _E, 0.0, 1.0, np.array([]), list(Distri), list(Cor),
                 Met=-1, Mett=mett, Recon=list(Recon)),
        dtype=float,
    )


def _put(pt, row, model, values, texts=()):
    pt.select_model(row, model)
    for col, value in enumerate(values):
        pt._value_input(row, col).setText(repr(float(value)))
    for col, text in dict(texts).items():
        pt._value_input(row, col).setText(text)


def _clear(pt):
    for row in range(1, len(pt.row_widgets)):
        pt.select_model(row, 'None')


@pytest.mark.gui
@pytest.mark.parametrize("mett", [0, 1], ids=["SMS", "CMS"])
def test_gui_chain_equals_hand_written_expansion(physics_app, mett):
    """Sextet + Distr(H) + Corr(delta) + Recon(eps) built as real table rows must
    equal the six Sextets it stands for, also built as table rows. Exercises
    read_model's flat layout for the marker rows (the placeholder slots, the
    Distri/Cor/Recon side lists), which the pure-physics test bypasses."""
    pt = physics_app.params_table

    _clear(pt)
    _put(pt, 1, 'Sextet', _SX)
    _put(pt, 2, 'Distr', [3, _GUI_LO, _GUI_HI, _GUI_NUM], texts={4: _GUI_PDF})
    _put(pt, 3, 'Corr', [1], texts={1: _GUI_CORR})
    _put(pt, 4, 'Recon', [2, _GUI_RLO, _GUI_RHI, len(_GUI_RW), 0, 0],
         texts={6: " ".join(repr(x) for x in _GUI_RW)})
    assert pt.get_conflicting_distr_targets() == []          # distinct par: 3, 1, 2
    y_chain = _table_spectrum(physics_app, mett)

    X = np.linspace(_GUI_LO, _GUI_HI, _GUI_NUM)
    ge = _eval(_GUI_PDF, X)
    ge = ge / ge.sum()
    XR = np.linspace(_GUI_RLO, _GUI_RHI, len(_GUI_RW))
    gr = np.asarray(_GUI_RW, float) / np.sum(_GUI_RW)
    corr_values = _eval(_GUI_CORR, X)

    _clear(pt)
    row = 1
    for i in range(len(_GUI_RW)):            # outer level = the LAST marker (Recon)
        for j in range(_GUI_NUM):
            one = list(_SX)
            one[0] = _SX[0] * gr[i] * ge[j]
            one[1] = corr_values[j]          # Corr target
            one[2] = XR[i]                   # Recon axis
            one[3] = X[j]                    # Distr axis
            _put(pt, row, 'Sextet', one)
            row += 1
    y_expand = _table_spectrum(physics_app, mett)

    depth = np.max(np.abs(1.0 - y_expand))
    assert np.max(np.abs(y_chain - y_expand)) / depth < 1e-9


@pytest.mark.gui
def test_duplicate_par_silently_discards_a_whole_distribution(physics_app):
    """Why get_conflicting_distr_targets has to block the run.

    Two Distr rows on one Sextet both aiming at par=3: the outer one writes
    pN[3] = X_outer, then the inner one overwrites it with X_inner. The outer
    axis is thrown away completely -- the result is the inner distribution alone
    -- while its Num still multiplies the number of copies computed. Nothing in
    the fit reports this, which is exactly why the table refuses it up front.
    """
    pt = physics_app.params_table
    inner = [3, 31.0, 35.0, 3]
    outer = [3, 20.0, 46.0, 3]

    _clear(pt)
    _put(pt, 1, 'Sextet', _SX)
    _put(pt, 2, 'Distr', inner, texts={4: "1+0*X"})
    _put(pt, 3, 'Distr', outer, texts={4: "1+0*X"})
    y_both = _table_spectrum(physics_app, 0)

    _clear(pt)
    _put(pt, 1, 'Sextet', _SX)
    _put(pt, 2, 'Distr', inner, texts={4: "1+0*X"})
    y_inner = _table_spectrum(physics_app, 0)

    _clear(pt)
    _put(pt, 1, 'Sextet', _SX)
    _put(pt, 2, 'Distr', outer, texts={4: "1+0*X"})
    y_outer = _table_spectrum(physics_app, 0)

    # The two rows do NOT combine: the result collapses onto the first one.
    assert np.allclose(y_both, y_inner, rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(y_both - y_outer)) > 1e-2

    # ...so the table must refuse it before a fit or a show-model ever starts.
    _clear(pt)
    _put(pt, 1, 'Sextet', _SX)
    _put(pt, 2, 'Distr', inner, texts={4: "1+0*X"})
    _put(pt, 3, 'Distr', outer, texts={4: "1+0*X"})
    assert [c['par'] for c in pt.get_conflicting_distr_targets()] == [3]
    assert physics_app.check_user_expressions("Show model") is False
