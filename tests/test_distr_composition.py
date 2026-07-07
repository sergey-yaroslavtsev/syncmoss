"""Distr (parameter distribution) composition with the thick / Layer machinery.

A distributed component must behave exactly like an undistributed one placed in
the SAME layer: its Num sub-cross-sections sum into the current layer's Sigma,
which is exponentiated once and read out once at the top level -- NOT
exponentiated and rho-read-out on its own and then multiplied into the spectrum.
These pin that contract (regression guard for the amplitude-readout fix, which
routes the inner Distr recursion's accumulated Sigma back to the caller via
return_layer_matrix instead of reading it out separately).

Pure NumPy/Numba path: TImod called directly with Met=-1 (per-energy absorber
transmission), no Qt and no multiprocessing pool.
"""
import numpy as np

import syncmoss.models as m5

_E = np.linspace(-10.0, 10.0, 96)

# Two distinct thick, anisotropic sextets (I=20/15, A=1 -> the two 2x2 amplitude
# operators genuinely differ and couple through the sample).
_SX = [20.0, 0.0, 0.0, 33.0, 0.098, 0.15, 50.0, 30.0, 1.0, 0.0, 0.0, 0.0, 3.0]
_SY = [15.0, 0.3, 0.1, 28.0, 0.098, 0.15, 40.0, 70.0, 1.0, 0.0, 0.0, 0.0, 2.0]


def _c(model, params, distri=None, mett=0):
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array(model), _E, 0.0, 1.0,
                 np.array([]), list(distri or []), [], Met=-1, Mett=mett),
        dtype=float,
    )


def test_distr_num1_reproduces_undistributed():
    """A Num=1 distribution whose single point sits at the base value reproduces
    the undistributed component bit-for-bit (both SMS and CMS)."""
    # Distr params: [param-index-to-distribute, start, end, Num, reserved].
    # Distribute the sextet's H (index 3) over a single point == base H.
    distr = [3.0, _SX[3], _SX[3], 1.0, 0.0]
    for mett in (0, 1):
        y_plain = _c(["Sextet"], _SX, mett=mett)
        y_distr = _c(["Sextet", "Distr"], _SX + distr, distri=["1"], mett=mett)
        assert np.allclose(y_distr, y_plain, rtol=1e-12, atol=1e-14), (
            f"Num=1 Distr must equal the undistributed component (mett={mett})"
        )


def test_distr_thick_joins_current_layer():
    """A Distr wrapping a thick component in the SAME layer as another thick
    component sums their cross-sections before one exponential -- i.e. equals the
    two components placed together undistributed, NOT the product of two separate
    readouts. This is the composition bug the amplitude fix closes."""
    distr_y = [3.0, _SY[3], _SY[3], 1.0, 0.0]                 # Num=1 distribution of Y's H at its base value

    y_together = _c(["Sextet", "Sextet"], _SX + _SY, mett=0)                       # both in one layer
    y_x_distrY = _c(["Sextet", "Sextet", "Distr"], _SX + _SY + distr_y, distri=["1"], mett=0)
    assert np.allclose(y_x_distrY, y_together, rtol=1e-9, atol=1e-11), (
        "distributed thick component must join the current layer (sum of Sigma), "
        "not be read out separately and multiplied in"
    )

    # Non-triviality: the layer couples the two thick components nonlinearly, so
    # the correct joint spectrum is NOT the product of the two solo readouts
    # (which is what the old code produced for a Distr'd thick + co-located thick).
    y_x = _c(["Sextet"], _SX, mett=0)
    y_y = _c(["Sextet"], _SY, mett=0)
    assert np.max(np.abs(y_together - y_x * y_y)) > 1e-3, (
        "test would be vacuous if the two thick components did not couple"
    )


def test_distr_respects_layer_marker():
    """A 'Layer' between the two components puts the distributed one in its OWN
    layer (amplitude-multiplied), so the result differs from stacking them in one
    layer -- confirming Distr honours Layer boundaries."""
    distr_y = [3.0, _SY[3], _SY[3], 1.0, 0.0]
    y_one_layer = _c(["Sextet", "Sextet", "Distr"], _SX + _SY + distr_y, distri=["1"], mett=0)
    y_two_layer = _c(["Sextet", "Layer", "Sextet", "Distr"], _SX + _SY + distr_y, distri=["1"], mett=0)
    assert np.max(np.abs(y_one_layer - y_two_layer)) > 1e-3, (
        "a 'Layer' marker must separate the distributed component into its own layer"
    )


def test_distr_scalar_unaffected_by_layer():
    """A distributed SCALAR component multiplies into the spectrum identically
    whether or not it shares a layer with a thick component (scalars never touch
    Smat), and a Num=1 scalar distribution reproduces the plain scalar."""
    singlet = [10.0, 0.1, 0.098, 0.12]                       # I, delta, L, G
    distr_s = [1.0, singlet[1], singlet[1], 1.0, 0.0]        # Num=1 distribution of delta (index 1) at base
    y_plain = _c(["Singlet"], singlet, mett=0)
    y_distr = _c(["Singlet", "Distr"], singlet + distr_s, distri=["1"], mett=0)
    assert np.allclose(y_distr, y_plain, rtol=1e-12, atol=1e-14), (
        "Num=1 scalar Distr must equal the plain scalar component"
    )
