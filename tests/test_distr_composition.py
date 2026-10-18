"""Distr (parameter distribution): the Num=1 identities.

A Num=1 distribution whose single point sits at the base value, with Corr
formulas that evaluate to the base values, must reproduce the undistributed
component EXACTLY, for every ordering of Distr and Corr markers and for a scalar
component as well. Genuine multi-point distributions -- including the layer
contract (the copies' cross-sections sum into the current layer's Sigma, a
'Layer' marker starts a new one) -- are checked against the model written out
by hand in test_distr_expansion.

Pure NumPy/Numba path: TImod called directly with Met=-1 (per-energy absorber
transmission), no Qt and no multiprocessing pool.
"""
import numpy as np
import pytest

import syncmoss.models as m5

_E = np.linspace(-10.0, 10.0, 96)

# A thick, anisotropic sextet (A=1, Am=1: its 2x2 amplitude operator carries the
# Faraday term). Am (index 9) sits right after A in the polarized Sextet.
_SX = [20.0, 0.0, 0.0, 33.0, 0.098, 0.15, 50.0, 30.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0]


def _c(model, params, distri=None, cor=None, mett=0):
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array(model), _E, 0.0, 1.0,
                 np.array([]), list(distri or []), list(cor or []), Met=-1, Mett=mett),
        dtype=float,
    )


def test_distr_scalar_num1_reproduces_the_plain_scalar():
    """A Num=1 distribution of a SCALAR component (it multiplies into the
    spectrum, never touches Smat) reproduces the plain scalar."""
    singlet = [10.0, 0.1, 0.098, 0.12]                       # I, delta, L, G
    distr_s = [1.0, singlet[1], singlet[1], 1.0, 0.0]        # Num=1 distribution of delta (index 1) at base
    y_plain = _c(["Singlet"], singlet, mett=0)
    y_distr = _c(["Singlet", "Distr"], singlet + distr_s, distri=["1"], mett=0)
    assert np.allclose(y_distr, y_plain, rtol=1e-12, atol=1e-14), (
        "Num=1 scalar Distr must equal the plain scalar component"
    )


# --- Multi-Distr / multi-Corr ORDERING (indexing) guards -------------------
# The Distr/Corr recursion tracks per-marker offsets (Dk/Ck walkback, mDk/mCk,
# and the Distri/Cor list slices). A Num=1 distribution whose single point sits
# at the base value, and a Corr formula that evaluates to the base value, must
# put every parameter back at its base -> reproduce the undistributed component
# EXACTLY, for every ordering of Distr and Corr markers. Distinct Corr constants
# also verify the Corr->param assignment isn't swapped.

def _D(idx, base):
    """Num=1 Distr of param `idx` pinned at its base value."""
    return [float(idx), base[idx], base[idx], 1.0, 0.0]


# (label, extra model markers after the Sextet, params after the Sextet, Distri, Cor)
_ORDERINGS = [
    ("D", ["Distr"], lambda b: _D(1, b), ["1"], []),
    ("DD", ["Distr", "Distr"], lambda b: _D(1, b) + _D(2, b), ["1", "1"], []),
    ("DDD", ["Distr", "Distr", "Distr"], lambda b: _D(1, b) + _D(2, b) + _D(3, b), ["1", "1", "1"], []),
    ("DC", ["Distr", "Corr"], lambda b: _D(1, b) + [2.0, 0.0], ["1"], None),          # Cor set below (needs base)
    ("DCC", ["Distr", "Corr", "Corr"], lambda b: _D(1, b) + [2.0, 0.0] + [3.0, 0.0], ["1"], None),
    ("DCD", ["Distr", "Corr", "Distr"], lambda b: _D(1, b) + [2.0, 0.0] + _D(3, b), ["1", "1"], None),
    ("DDC", ["Distr", "Distr", "Corr"], lambda b: _D(1, b) + _D(2, b) + [3.0, 0.0], ["1", "1"], None),
    ("DCDC", ["Distr", "Corr", "Distr", "Corr"],
     lambda b: _D(1, b) + [2.0, 0.0] + _D(3, b) + [4.0, 0.0], ["1", "1"], None),
]


def _cor_consts(label, base):
    """Constant Corr formulas that pin their target param at its base value,
    in model order, matching the [2.0,0.0]/[3.0,0.0]/[4.0,0.0] Corr rows above."""
    idx_by_label = {"DC": [2], "DCC": [2, 3], "DCD": [2], "DDC": [3], "DCDC": [2, 4]}
    return [str(base[i]) for i in idx_by_label.get(label, [])]


@pytest.mark.parametrize("label,markers,make_params,distri,_", _ORDERINGS)
def test_num1_ordering_reproduces_base(label, markers, make_params, distri, _):
    """Every Distr/Corr ordering, at Num=1 pinned to base values, reproduces the
    undistributed Sextet exactly -- for both SMS and CMS (no index/offset drift)."""
    cor = _cor_consts(label, _SX)
    model = ["Sextet"] + markers
    params = _SX + make_params(_SX)
    for mett in (0, 1):
        base = _c(["Sextet"], _SX, mett=mett)
        y = _c(model, params, distri=distri, cor=cor, mett=mett)
        assert np.allclose(y, base, rtol=1e-11, atol=1e-13), (
            f"ordering {label!r} (mett={mett}) did not reproduce the base component"
        )
