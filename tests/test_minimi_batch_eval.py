"""minimi_hi's ``batch_eval`` (the SYNCtime-branch hook): the Jacobian's bumped
vectors are handed over as ONE list, which changes neither the evaluations nor
the fit -- the result is bit-identical to the one-by-one path. And its
``jacobian`` (the caller's derivatives instead of differences): the same fit as
the differences give, links followed by the chain rule, expressions in the
caller's rows (total derivatives). And its ``refresh_jacobian``: a step that
fails with the Jacobian of an earlier point is tried again with the one of its
own point -- the same fit in fewer evaluations."""
import numpy as np
import pytest

import syncmoss.minimi_lib as mi


def _model(x, p):
    return p[0] - p[1] * np.exp(-(x - p[3]) ** 2 / (2 * p[2] ** 2))


def _derivatives(x, p):
    """d_model/d_p of _model, one row per parameter (a fifth, unused one: zeros)."""
    g = np.exp(-(x - p[3]) ** 2 / (2 * p[2] ** 2))
    rows = [np.ones_like(x), -g, -p[1] * g * (x - p[3]) ** 2 / p[2] ** 3, -p[1] * g * (x - p[3]) / p[2] ** 2]
    return np.array(rows + [np.zeros_like(x)] * (len(p) - 4))


def _two_dips(x, p):
    """_model plus a second dip of amplitude p[4] at p[3] + 2."""
    return _model(x, p[:4]) - p[4] * np.exp(-(x - p[3] - 2) ** 2 / (2 * p[2] ** 2))


def _two_dips_derivatives(x, p):
    rows = _derivatives(x, p)
    g2 = np.exp(-(x - p[3] - 2) ** 2 / (2 * p[2] ** 2))
    rows[2] += -p[4] * g2 * (x - p[3] - 2) ** 2 / p[2] ** 3
    rows[3] += -p[4] * g2 * (x - p[3] - 2) / p[2] ** 2
    rows[4] = -g2
    return rows


@pytest.mark.parametrize('weights', ['P', 'W'])
def test_an_analytic_jacobian_gives_the_fit_the_differences_give(weights):
    x = np.linspace(-5.0, 5.0, 200)
    y = np.random.default_rng(3).poisson(_model(x, [1000.0, 50.0, 0.7, 0.3])).astype(float)
    p0 = np.array([950.0, 40.0, 0.8, 0.2])
    kw = dict(MI=40, MI2=10, nu0=2.618, tau0=1e-3, eps=1e-9, weights=weights)
    plain = mi.minimi_hi(_model, x, y, p0.copy(), **kw)
    analytic = mi.minimi_hi(_model, x, y, p0.copy(), jacobian=lambda p: _derivatives(x, p), **kw)
    assert not np.allclose(plain[0], p0)                              # the fit moved
    assert np.all(np.abs(analytic[0] - plain[0]) < 1e-3 * plain[1])  # the same optimum, within its errors
    assert np.allclose(analytic[1], plain[1], rtol=1e-3)


def test_an_analytic_jacobian_follows_a_link():
    """p[4] = 0.5 p[1] (confu): the link's target moves with its source."""
    x = np.linspace(-5.0, 5.0, 300)
    y = np.random.default_rng(5).poisson(_two_dips(x, [1000.0, 60.0, 0.5, -1.0, 30.0])).astype(float)
    p0 = np.array([950.0, 50.0, 0.6, -0.9, 25.0])
    confu = np.array([[4], [1], [0.5]], dtype=float)
    kw = dict(fix=np.array([4]), confu=confu, MI=40, MI2=10, nu0=2.618, tau0=1e-3, eps=1e-9, weights='P')

    def at_link(p):
        return np.concatenate([p[:4], [0.5 * p[1]]])
    plain = mi.minimi_hi(lambda _x, p: _two_dips(_x, at_link(p)), x, y, p0.copy(), **kw)
    analytic = mi.minimi_hi(lambda _x, p: _two_dips(_x, at_link(p)), x, y, p0.copy(),
                            jacobian=lambda p: _two_dips_derivatives(x, at_link(p)), **kw)
    assert not np.allclose(plain[0][:4], p0[:4])                                  # the fit moved
    assert np.all(np.abs(analytic[0][:4] - plain[0][:4]) < 1e-3 * plain[1][:4])
    # the rows themselves: the chain rule gives the differences' columns; without the link, p[1]'s is short
    model = lambda _x, p: _two_dips(_x, at_link(p))                               # noqa: E731
    base = model(x, p0)
    diff, _ = mi.compute_jacobian(model, p0, x, kw['fix'], confu, base, [], np.array([], dtype=int))
    rows = lambda p: _two_dips_derivatives(x, at_link(p))                         # noqa: E731
    chain, _ = mi.compute_jacobian(model, p0, x, kw['fix'], confu, base, [], np.array([], dtype=int), jacobian=rows)
    none, _ = mi.compute_jacobian(model, p0, x, kw['fix'], np.array([[-1], [-1], [0]], dtype=float), base, [],
                                  np.array([], dtype=int), jacobian=rows)
    scale = np.abs(diff).max(axis=1, keepdims=True)
    assert np.allclose(chain / scale, diff / scale, atol=1e-4)
    assert not np.allclose(none / scale, diff / scale, atol=1e-4)


def test_an_analytic_jacobian_with_an_expression_is_of_total_derivatives():
    """p[4] = p[1]**2 / 120 (an Expression): the caller's row of p[1] holds what
    the expression moves, and the fit ends where the differences end."""
    x = np.linspace(-5.0, 5.0, 300)
    y = np.random.default_rng(5).poisson(_two_dips(x, [1000.0, 60.0, 0.5, -1.0, 30.0])).astype(float)
    p0 = np.array([950.0, 50.0, 0.6, -0.9, 50.0 ** 2 / 120])
    kw = dict(fix=np.array([4]), Expr=['p[1]**2 / 120'], NExpr=np.array([4]), MI=40, MI2=10, nu0=2.618, tau0=1e-3,
              eps=1e-9, weights='P')

    def total(p):
        q = np.concatenate([p[:4], [p[1] ** 2 / 120]])
        rows = _two_dips_derivatives(x, q)
        rows[1] += q[1] / 60 * rows[4]
        rows[4] = 0
        return rows
    plain = mi.minimi_hi(_two_dips, x, y, p0.copy(), **kw)
    analytic = mi.minimi_hi(_two_dips, x, y, p0.copy(), jacobian=total, **kw)
    assert not np.allclose(plain[0][:4], p0[:4])                                  # the fit moved
    assert np.all(np.abs(analytic[0][:4] - plain[0][:4]) < 1e-3 * plain[1][:4])
    assert np.allclose(analytic[1][:4], plain[1][:4], rtol=1e-3)
    assert analytic[0][4] == analytic[0][1] ** 2 / 120                            # the expression holds
    # the rows themselves: the differences' columns; without the expression's part, p[1]'s is short
    no_links = np.array([[-1], [-1], [0]], dtype=float)
    args = (_two_dips, p0, x, kw['fix'], no_links, _two_dips(x, p0), kw['Expr'], kw['NExpr'])
    diff, _ = mi.compute_jacobian(*args)
    rows, _ = mi.compute_jacobian(*args, jacobian=total)
    short, _ = mi.compute_jacobian(*args, jacobian=lambda p: _two_dips_derivatives(x, p))
    scale = np.abs(diff).max(axis=1, keepdims=True)
    assert np.allclose(rows / scale, diff / scale, atol=1e-4)
    assert not np.allclose(short / scale, diff / scale, atol=1e-4)


@pytest.mark.parametrize('weights', ['P', 'W'])
def test_a_failed_step_takes_the_jacobian_of_its_own_point(weights):
    """refresh_jacobian: the same optimum as without, in fewer model evaluations
    (the steps that failed with an old Jacobian: 26 instead of 99 here), never
    two Jacobians at one point; the same with differences, and with a
    parameter pushed onto its bound (fixed there, then rechecked)."""
    x = np.linspace(-5.0, 5.0, 200)
    y = np.random.default_rng(3).poisson(_model(x, [1000.0, 50.0, 0.7, 0.3])).astype(float)
    p0 = np.array([950.0, 40.0, 0.8, 0.2])
    kw = dict(MI=40, MI2=10, nu0=2.618, tau0=1e-3, eps=1e-9, weights=weights)

    def run(refresh, **extra):
        evaluations, points = [0], []

        def model(_x, p):
            evaluations[0] += 1
            return _model(_x, p)

        def rows(p):
            points.append(np.array(p, dtype=float))
            return _derivatives(x, p)
        fit = mi.minimi_hi(model, x, y, p0.copy(), jacobian=rows, refresh_jacobian=refresh, **kw, **extra)
        return fit, evaluations[0], points

    (old, n_old, _), (new, n_new, points) = run(False), run(True)
    assert not np.allclose(old[0], p0)                                    # the fit moved
    assert np.all(np.abs(new[0] - old[0]) < 1e-3 * old[1])               # the same optimum, within its errors
    assert new[2] == pytest.approx(old[2], rel=1e-9)
    assert n_new < n_old / 2
    assert len(points) > 2 and not any(np.array_equal(a, b) for a, b in zip(points, points[1:]))
    differences = mi.minimi_hi(_model, x, y, p0.copy(), refresh_jacobian=True, **kw)
    assert np.all(np.abs(differences[0] - old[0]) < 1e-3 * old[1])
    bounds = np.array([[-np.inf] * 4, [np.inf, 45.0, np.inf, np.inf]])  # the depth held below its 50
    (held, _, _), (held_new, _, _) = run(False, bounds=bounds), run(True, bounds=bounds)
    assert held[0][1] == held_new[0][1] == 45.0
    assert np.allclose(held_new[0], held[0], rtol=1e-5)


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
