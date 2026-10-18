"""Recon (model-independent distribution reconstruction) composition + the
smoothness-regularization helpers.

A Recon reuses the exact replication machinery of Distr, but its per-channel
density is a vector of FREE fit weights (carried in the parallel ``Recon`` list)
instead of an evaluated PDF expression. These tests pin:

* the forward-model contract — a Recon joins the current layer like a Distr, a
  'Layer' marker separates it, and every ordering with Distr/Corr reproduces the
  undistributed component when Num=1 with a single weight (genuine multi-point
  Recon chains are checked against the hand-written model in
  test_distr_expansion), and
* the fit-side regularization helpers (pure math, no numba / no pool): the
  D->lambda map, and that the first/second-difference penalties vanish for a flat
  density and for D=0, and grow with roughness / D.

Pure NumPy/Numba path for the physics part: TImod called directly with Met=-1,
no Qt and no multiprocessing pool (as in test_distr_composition).
"""
import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.fitting_io import (
    recon_fit_layout, _recon_penalty_rows, _recon_penalty_length, _recon_lambda,
)

_E = np.linspace(-10.0, 10.0, 96)

# Two distinct thick, anisotropic sextets (the same as in test_distr_expansion).
_SX = [20.0, 0.0, 0.0, 33.0, 0.098, 0.15, 50.0, 30.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0]
_SY = [15.0, 0.3, 0.1, 28.0, 0.098, 0.15, 40.0, 70.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0]


def _c(model, params, distri=None, cor=None, recon=None, mett=0):
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array(model), _E, 0.0, 1.0,
                 np.array([]), list(distri or []), list(cor or []),
                 Met=-1, Mett=mett, Recon=list(recon or [])),
        dtype=float,
    )


def _R1(idx, base):
    """A Num=1 Recon block for param ``idx`` pinned at its base value (7 slots:
    par, L, R, Num, D_dif, D_dif2, weight-placeholder)."""
    return [float(idx), base[idx], base[idx], 1.0, 0.0, 0.0, 0.0]


# --- forward-model contract -------------------------------------------------

def test_recon_thick_joins_current_layer():
    """A Recon wrapping a thick component in the SAME layer as another thick
    component sums their cross-sections before one exponential (like Distr)."""
    recon_y = _R1(3, _SY)
    y_together = _c(["Sextet", "Sextet"], _SX + _SY, mett=0)
    y_x_reconY = _c(["Sextet", "Sextet", "Recon"], _SX + _SY + recon_y,
                    recon=[np.array([1.0])], mett=0)
    assert np.allclose(y_x_reconY, y_together, rtol=1e-9, atol=1e-11), (
        "distributed (Recon) thick component must join the current layer"
    )


def test_recon_respects_layer_marker():
    recon_y = _R1(3, _SY)
    y_one_layer = _c(["Sextet", "Sextet", "Recon"], _SX + _SY + recon_y,
                     recon=[np.array([1.0])], mett=0)
    y_two_layer = _c(["Sextet", "Layer", "Sextet", "Recon"], _SX + _SY + recon_y,
                     recon=[np.array([1.0])], mett=0)
    assert np.max(np.abs(y_one_layer - y_two_layer)) > 1e-3, (
        "a 'Layer' marker must separate the Recon component into its own layer"
    )


# --- mixed Distr / Corr / Recon ORDERING (indexing) guards ------------------
# Num=1 pinned to base + base-valued Corr must reproduce the undistributed Sextet
# for every interleaving of Distr, Corr and Recon markers -> the Dk/Ck/Rk walk-back
# and the Distri/Cor/Recon list slices stay in lockstep.

def _D1(idx, base):
    return [float(idx), base[idx], base[idx], 1.0, 0.0]


# (label, markers, params-after-Sextet, distri, cor, recon)
_ORDERINGS = [
    ("R",   ["Recon"],                 lambda b: _R1(1, b),                       [],       [],            [np.array([1.0])]),
    ("RR",  ["Recon", "Recon"],        lambda b: _R1(1, b) + _R1(2, b),           [],       [],            [np.array([1.0]), np.array([1.0])]),
    ("DR",  ["Distr", "Recon"],        lambda b: _D1(1, b) + _R1(2, b),           ["1"],    [],            [np.array([1.0])]),
    ("RD",  ["Recon", "Distr"],        lambda b: _R1(1, b) + _D1(2, b),           ["1"],    [],            [np.array([1.0])]),
    ("RC",  ["Recon", "Corr"],         lambda b: _R1(1, b) + [2.0, 0.0],          [],       None,          [np.array([1.0])]),
    ("RCR", ["Recon", "Corr", "Recon"],lambda b: _R1(1, b) + [2.0, 0.0] + _R1(3, b), [],    None,          [np.array([1.0]), np.array([1.0])]),
]


def _cor_consts(label, base):
    idx_by_label = {"RC": [2], "RCR": [2]}
    return [str(base[i]) for i in idx_by_label.get(label, [])]


@pytest.mark.parametrize("label,markers,make_params,distri,_c_unused,recon", _ORDERINGS)
def test_recon_num1_ordering_reproduces_base(label, markers, make_params, distri, _c_unused, recon):
    cor = _cor_consts(label, _SX)
    model = ["Sextet"] + markers
    params = _SX + make_params(_SX)
    for mett in (0, 1):
        base = _c(["Sextet"], _SX, mett=mett)
        y = _c(model, params, distri=distri, cor=cor, recon=recon, mett=mett)
        assert np.allclose(y, base, rtol=1e-11, atol=1e-13), (
            f"ordering {label!r} (mett={mett}) did not reproduce the base component"
        )


# --- regularization helpers (pure math, no numba / no pool) -----------------

def test_recon_lambda_endpoints():
    assert _recon_lambda(0.0) == 0.0
    assert _recon_lambda(0.5) == pytest.approx(1.0, rel=1e-3)
    assert _recon_lambda(1.0) > 1e5                 # D=1 reachable and strongly forcing
    assert np.isfinite(_recon_lambda(1.0))          # ...but finite (no division by zero)


def test_recon_penalty_flat_and_zero_D():
    infos = [{'off': 8, 'num': 5, 'd1': 0.5, 'd2': 0.5, 're': 0, 'wstart': 0}]
    flat = np.ones(5)
    spiky = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # A flat density has zero 1st and 2nd differences -> zero penalty.
    assert np.allclose(_recon_penalty_rows(flat, infos, 1.0), 0.0)
    # A spiky density is penalized.
    assert np.sum(_recon_penalty_rows(spiky, infos, 1.0) ** 2) > 0.0
    # D_dif == D_dif2 == 0 -> unconstrained (no penalty), even for a spike.
    infos0 = [{'off': 8, 'num': 5, 'd1': 0.0, 'd2': 0.0, 're': 0, 'wstart': 0}]
    assert np.allclose(_recon_penalty_rows(spiky, infos0, 1.0), 0.0)


def test_recon_penalty_grows_with_D():
    infos_lo = [{'off': 8, 'num': 6, 'd1': 0.2, 'd2': 0.0, 're': 0, 'wstart': 0}]
    infos_hi = [{'off': 8, 'num': 6, 'd1': 0.9, 'd2': 0.0, 're': 0, 'wstart': 0}]
    spiky = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    lo = np.sum(_recon_penalty_rows(spiky, infos_lo, 1.0) ** 2)
    hi = np.sum(_recon_penalty_rows(spiky, infos_hi, 1.0) ** 2)
    assert hi > lo > 0.0


def test_recon_penalty_length():
    infos = [{'num': 5}, {'num': 3}]
    # num=5 -> (5-1)+(5-2)=7 ; num=3 -> (3-1)+(3-2)=3 ; total 10
    assert _recon_penalty_length(infos) == 10


def test_recon_regularized_reconstruction_via_minimizer():
    """End-to-end mechanism check of the fit path WITHOUT the GUI: drive
    minimi_hi with an augmented model (data rows + √λ·difference penalty rows from
    the real _recon_penalty_rows helper) and n_reg, exactly as fit_single_spectrum
    does for a Recon. A smooth true density fed through a well-conditioned linear
    response must be recovered, the covariance must stay finite (no divide-by-≈0
    from the zero-target penalty rows), and weights must respect the >=0 bound.
    """
    import syncmoss.minimi_lib as mi

    num = 15
    ch = np.arange(num)
    w_true = np.exp(-0.5 * ((ch - 7.0) / 2.0) ** 2)          # a smooth bump

    # Well-conditioned linear response: M distinct smooth channel profiles.
    M = 48
    t = np.linspace(0.0, 1.0, M)
    K = np.stack([np.exp(-0.5 * ((t - c / (num - 1)) / 0.16) ** 2) for c in ch], axis=1)  # (M, num)
    data = K @ w_true

    infos = [{'off': 0, 'num': num, 'd1': 0.3, 'd2': 0.0, 're': 0, 'wstart': 0}]
    reg_scale = float(np.sum((data - data.mean()) ** 2)) or 1.0
    n_reg = _recon_penalty_length(infos)

    def model_func(_x, w):
        spec = K @ w
        pen = _recon_penalty_rows(w, infos, reg_scale)
        return np.concatenate([spec, pen])

    x_aug = np.concatenate([t, np.zeros(n_reg)])
    y_aug = np.concatenate([data, np.zeros(n_reg)])
    w0 = np.full(num, 1.0 / num)
    bounds = np.array([[0.0] * num, [np.inf] * num], dtype=float)

    w_fit, er, chi2, cov, _pearson = mi.minimi_hi(model_func, x_aug, y_aug, w0,
                                                  bounds=bounds, n_reg=n_reg, MI=40, MI2=20)

    assert np.all(np.isfinite(cov)), "penalty rows must not poison the covariance"
    assert np.all(w_fit >= -1e-9), "weights must respect the >=0 bound"
    # The data is reproduced by the reconstruction.
    assert np.max(np.abs(K @ w_fit - data)) < 5e-2
    # A smooth truth is recovered reasonably (mild regularization, exact data).
    assert np.corrcoef(w_fit, w_true)[0, 1] > 0.95


def test_recon_fit_layout():
    # baseline(8) + Sextet(14) + Recon(7)
    model = ['Sextet', 'Recon']
    p = np.zeros(8 + 14 + 7)
    off = 8 + 14
    p[off + 3] = 5      # Num
    p[off + 4] = 0.3    # D_dif
    p[off + 5] = 0.7    # D_dif2
    infos, n_weights = recon_fit_layout(model, p)
    assert n_weights == 5
    assert len(infos) == 1
    assert infos[0]['off'] == off
    assert infos[0]['num'] == 5
    assert infos[0]['d1'] == pytest.approx(0.3)
    assert infos[0]['d2'] == pytest.approx(0.7)
    assert infos[0]['wstart'] == 0
