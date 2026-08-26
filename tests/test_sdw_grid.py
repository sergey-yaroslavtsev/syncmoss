"""Grid optimisation of the SCDW (spin/charge density wave) line shapes.

``models.SDW_thick_terms`` bins each line's ``Num`` wave positions onto a grid
tied to the Voigt width, so the Voigt count is Num-independent. It must match the
reference direct sum ``models.SDW_thick_terms_direct`` (the ground truth), and it
must preserve the two physics invariants of the branch: the single-sextet edge
case and the balanced-wave Faraday cancellation.

Pure NumPy/Numba path -- no Qt needed.
"""
import numpy as np
import pytest

import syncmoss.models as m5

_E = np.linspace(-8.0, 8.0, 512)
_WL, _WG = 0.15, 0.15
_MULCO = 1.0            # tests work in unscaled (MulCo=1) units
_STEPS = 12.0           # explicit grid resolution (the per-component 'N/Γ' knob)

# A representative wave: base field (so the Faraday survives), several odd/even
# harmonics, both field->shift correlations and a CDW phase. hodd/dev are arrays
# (the jitted SDW_thick_terms/_direct take float64 arrays).
_WAVE = dict(d0=0.1, eps0=0.05, KeH=0.005, H0=30.0,
             hodd=np.array([5.0, 1.5, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0]), phi_deg=25.0,
             KdH=0.003, dev=np.array([0.03, 0.01, 0.0, 0.0]))


def _L_direct(num, **wave):
    v = m5.SDW_thick_terms_direct(Num=num, **wave)
    return np.array([(m5.Voight_c(_WL, _WG, _E[:, None] - np.asarray(vk)[None, :])).sum(axis=1) / len(vk)
                     for vk in v])


def _L_grid(num, **wave):
    grids, wts = m5.SDW_thick_terms(Num=num, WL=_WL, WG=_WG, MulCo=_MULCO, steps=_STEPS, **wave)
    return np.array([(m5.Voight_c(_WL, _WG, _E[:, None] - g[None, :]) * w[None, :]).sum(axis=1)
                     for g, w in zip(grids, wts)])


def test_grid_matches_direct_reference():
    truth = _L_direct(6000, **_WAVE)                 # dense direct sum = ground truth
    got = _L_grid(500, **_WAVE)
    rel = np.max(np.abs(got - truth)) / np.max(np.abs(truth))
    assert rel < 3.0e-3, f"grid deviates from direct reference by {rel:.2e} of peak"


def test_grid_weights_are_normalised():
    _, wts = m5.SDW_thick_terms(Num=500, WL=_WL, WG=_WG, MulCo=_MULCO, steps=_STEPS, **_WAVE)
    for k, w in enumerate(wts):
        assert w.sum() == pytest.approx(1.0, abs=1e-12), f"line {k} weights sum to {w.sum()}"


def test_grid_edge_case_is_single_sextet():
    # No harmonics, no correlations, H0 != 0 -> every line sits at ONE position,
    # so each grid collapses to a single node (a plain sextet).
    grids, wts = m5.SDW_thick_terms(d0=0.1, eps0=0.05, KeH=0.0, H0=30.0,
                                    hodd=np.zeros(8), phi_deg=0.0, KdH=0.0, dev=np.zeros(4),
                                    Num=500, WL=_WL, WG=_WG, MulCo=_MULCO, steps=_STEPS)
    for g, w in zip(grids, wts):
        assert len(g) == 1 and w[0] == pytest.approx(1.0)


def test_grid_balanced_wave_faraday_cancellation():
    # Balanced wave (H0 = 0, KeH = KdH = 0): +/- sites populate equally, so the
    # position distributions of lines (1,6) and (3,4) coincide -> L1 == L6,
    # L3 == L4 and the resolved Faraday terms cancel in the branch.
    bal = dict(d0=0.1, eps0=0.0, KeH=0.0, H0=0.0,
               hodd=np.array([5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), phi_deg=0.0,
               KdH=0.0, dev=np.zeros(4))
    L = _L_grid(500, **bal)
    assert np.allclose(L[0], L[5], atol=1e-9), "balanced wave: L1 != L6"
    assert np.allclose(L[2], L[3], atol=1e-9), "balance wave: L3 != L4"
