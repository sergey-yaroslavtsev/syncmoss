"""Golden / regression tests for every fittable model (scalar, thick, and combos).

These pin the exact spectrum produced by ``models.TImod`` for fixed, reasonable
(non-edge) parameters of each model. They are the safety net for refactors of the
physics kernels (e.g. splitting Ham_mono / relax_MS into shared core + thin/thick
intensity helpers): the numbers must not move.

The reference values in ``golden_models.json`` were produced from the validated
implementation; regenerate that file deliberately (and review the diff) only when
a physics change is intended.

Pure NumPy/Numba path — no Qt needed.
"""
import json
import os

import numpy as np
import pytest

import syncmoss.models as m5

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_models.json")
with open(_GOLDEN_PATH) as _f:
    _GOLDEN = json.load(_f)

_E = np.linspace(_GOLDEN["E_lo"], _GOLDEN["E_hi"], _GOLDEN["E_n"])


def _compute(model, params):
    """Recompute a model's spectrum the same way the golden file was generated.

    Met=-1 is the per-energy ("recursive") entry point: V starts at 0 (no
    baseline) and E is taken verbatim, so the model parameters start at index 0.
    Mett=0 selects the single-line (SMS, Ham_mono) Hamiltonian path.

    The complex-Voigt method and dispersion sign are pinned here to the shipping
    defaults so the golden values are reproducible regardless of any developer
    switch of ``models.COMPLEX_VOIGT_METHOD`` (e.g. while comparing 'wofz' vs
    'pseudo'); the golden file was generated with exactly these settings.
    """
    m5.COMPLEX_VOIGT_METHOD = 'pseudo'
    m5.DISPERSION_SIGN = +1.0
    return np.asarray(
        m5.TImod(_E, np.array(params, float), np.array(model), _E, 0.0, 1.0,
                 np.array([]), [], [], Met=-1, Mett=0),
        dtype=float,
    )


@pytest.mark.parametrize("name", sorted(_GOLDEN["cases"].keys()))
def test_model_matches_golden(name):
    case = _GOLDEN["cases"][name]
    got = _compute(case["model"], case["params"])
    expected = np.asarray(case["values"], dtype=float)
    assert got.shape == expected.shape, f"{name}: shape {got.shape} != {expected.shape}"
    assert np.all(np.isfinite(got)), f"{name}: non-finite values"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-8), (
        f"{name}: spectrum drifted from golden reference "
        f"(max abs diff {np.max(np.abs(got - expected)):.3e})"
    )
