"""minimi_hi's ``batch_eval`` (the SYNCtime-branch hook): the Jacobian's bumped
vectors are handed over as ONE list, which changes neither the evaluations nor
the fit -- the result is bit-identical to the one-by-one path."""
import numpy as np

import syncmoss.minimi_lib as mi


def _model(x, p):
    return p[0] - p[1] * np.exp(-(x - p[3]) ** 2 / (2 * p[2] ** 2))


def test_a_batched_jacobian_gives_the_same_fit_bit_for_bit():
    x = np.linspace(-5.0, 5.0, 200)
    y = np.random.default_rng(3).poisson(_model(x, [1000.0, 50.0, 0.7, 0.3])).astype(float)
    p0 = np.array([950.0, 40.0, 0.8, 0.2])
    kw = dict(MI=20, MI2=10, nu0=2.618, tau0=1e-3, eps=1e-6)
    batches = []

    def batch(plist):
        batches.append(len(plist))
        return [_model(x, q) for q in plist]

    plain = mi.minimi_hi(_model, x, y, p0.copy(), **kw)
    batched = mi.minimi_hi(_model, x, y, p0.copy(), batch_eval=batch, **kw)
    assert np.array_equal(plain[0], batched[0])
    assert np.array_equal(plain[1], batched[1])
    assert plain[2] == batched[2]
    assert batches and all(n == 4 for n in batches)     # every Jacobian: its 4 columns, together
