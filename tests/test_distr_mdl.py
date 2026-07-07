"""End-to-end validation of multi-Distr and Distr+Corr models against the
analytical component they mimic, using real example model files.

Each .mdl holds TWO components in the same section:
  * Distr_distr.mdl -- a Sextet with intrinsic Gaussian line width (WG) and a
    hyperfine-field spread (GH), followed by a second Sextet that mimics BOTH via
    two distributions (distr + distr: a Gaussian spread of the shift and of H).
  * Distr_corr.mdl  -- an MDGD (built-in parameter-correlated Gaussian spreads)
    followed by a Sextet that mimics it via one distribution + one correlation
    (distr + corr).

The distributed twin should reproduce the analytical component (up to the finite
number of distribution points). We isolate each component with the I=0 trick
(zeroing the other component's intensity leaves its transmission == 1) and
compare the two absorber spectra computed through the REAL forward pipeline
(``models.TI`` at Met=0). This exercises the multi-Distr / Distr+Corr recursion
(the Dk/Ck offset bookkeeping, the ge weighting, and the Distri/Cor list slices)
on genuine models -- a guard against ordering/indexing regressions.
"""
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss.constants import number_of_baseline_parameters as NB
from syncmoss.model_io import load_model_from_path, mod_len_def, read_model

_DATA = os.path.join(os.path.dirname(__file__), "data")
_X = np.linspace(-12.0, 12.0, 400)
_INS = np.array([0.1, 0.0, 1.0])     # one narrow source line (Met=0 instrumental triple)


class _SerialPool:
    """Minimal drop-in for the multiprocessing pool TI expects (single process)."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def _prepare(app, path):
    """Load a .mdl into the app and return (model, effective_p, Distri, Cor).

    Applies the Expression and constraint passes exactly as the compute worker
    does before calling TI (expressions first, then constraints)."""
    load_model_from_path(app, path)
    model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, DistriN = read_model(app)
    p = np.array(p, dtype=float)
    ns = vars(m5)
    for i in range(len(NExpr)):
        p[int(NExpr[i])] = eval(Expr[i], ns, {"p": p})     # noqa: S307 - model-defined expressions
    for i in range(len(con1)):
        p[int(con1[i])] = p[int(con2[i])] * con3[i]
    return model, p, list(Distri), list(Cor)


def _ti(model, p, Distri, Cor):
    return np.asarray(
        m5.TI(_X, p, model, 30, _SerialPool(), 0.0, 1.0, _INS, Distri, Cor, Met=0, Norm=1),
        dtype=float,
    )


def _depth(y):
    """Absorption depth fraction (0 = no absorption); baseline = the spectrum max."""
    top = np.max(y)
    return (top - y) / (top + 1e-12)


# (file, tolerance on depth-normalised max|ref-dist|)
_MDL_CASES = [
    ("Distr_distr.mdl", 1.0e-3),   # measured ~1.6e-4
    ("Distr_corr.mdl", 4.0e-3),    # measured ~1.0e-3
]


@pytest.mark.parametrize("fname,tol", _MDL_CASES)
def test_distributed_twin_matches_analytical_component(physics_app, fname, tol):
    path = os.path.join(_DATA, fname)
    if not os.path.exists(path):
        pytest.skip(f"missing model file {fname}")

    model, p, Distri, Cor = _prepare(physics_app, path)
    # The first two entries are the analytical reference and its distributed twin.
    assert len(model) >= 2 and "Distr" in model, f"{fname}: unexpected model {model}"
    i_ref = NB                                                # reference intensity
    i_dist = NB + mod_len_def(model[0], include_special=False)  # distributed-twin intensity

    full = _ti(model, p, Distri, Cor)
    assert np.all(np.isfinite(full)), f"{fname}: full spectrum non-finite"

    p_ref = p.copy(); p_ref[i_dist] = 0.0                     # kill the twin -> analytical only
    p_dist = p.copy(); p_dist[i_ref] = 0.0                    # kill the reference -> distributed only
    ref = _ti(model, p_ref, Distri, Cor)
    dist = _ti(model, p_dist, Distri, Cor)

    d_ref, d_dist = _depth(ref), _depth(dist)
    # Both must be genuine, comparable absorption features (not a vacuous match).
    assert d_ref.max() > 0.01 and d_dist.max() > 0.01, f"{fname}: absorption too shallow to test"
    assert abs(d_ref.max() - d_dist.max()) / d_ref.max() < 0.1, f"{fname}: depths differ too much"
    # The distributed twin reproduces the analytical component to < tol of depth.
    worst = float(np.max(np.abs(d_ref - d_dist)))
    assert worst < tol, (
        f"{fname}: distributed twin deviates from analytical component by {worst:.2e} "
        f"(depth-normalised, tol {tol:.1e}) -- possible Distr/Corr ordering regression"
    )
