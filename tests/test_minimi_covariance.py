"""minimi_hi's weights, errors and chi-squares (docs/minimization.md).

Unweighted ('U', the default) its covariance is the sandwich H^-1 J V J^T H^-1
(H = J J^T, V = |y|+1); weighted ('W': 1/(|y|+1), 'P': the Poisson likelihood,
or an array) it is (J W J^T)^-1; times Pearson's reduced chi-square (none
where the model is not counts) unless chi2_correction=False. The block
the sandwich replaced weighted every point by 1/model and inverted with a numba
Gauss-Jordan kernel, which

* made the covariance indefinite when the model is not counts (Calibration
  fits velocities, of both signs): a negative variance, "invalid value
  encountered in sqrt", nan for that parameter;
* could not pivot through the blocks of zeros of spectra that share no
  parameter: "error in fast calculation of inverse matrix", then numpy;
* gave nan or 0 by the luck of rounding when two parameters only act together,
  even for well-determined ones -- now the later of them is reported like a
  fixed one;
* read the errors off the final fix list while the covariance kept the rows of
  the last Jacobian, so after a late bound hit the next parameter showed the
  error of the one on the bound;
* was the covariance of a WEIGHTED fit: 40 % too small on counts spanning
  decades;
* came back without the sqrt(chi2) the errors have, so an Expression or a %
  intensity propagated through it was 1/sqrt(chi2) of what the errors say.
"""
import inspect
import warnings

import numpy as np
import pytest

from syncmoss import minimi_lib as mi
from syncmoss.support_math import calculate_expression_error

X = np.linspace(-6, 6, 256)
N_CH = 512
LINE_V = np.array([-5.312, -3.076, -0.840, 0.840, 3.076, 5.312])     # alpha-Fe, mm/s


def _drive(ch, p):
    """Calibration's sinusoidal drive: the velocity at channel ch."""
    return (p[0] / np.pi * N_CH * np.sin(np.pi / N_CH)
            * np.cos(p[1] + np.pi / N_CH * (2 * ch + 1)) + p[2])


def _line_channels():
    v = _drive(np.arange(N_CH, dtype=float), [-10.0, 0.3, 0.05])
    half = N_CH // 2
    return np.array([start + np.abs(v[start:start + half] - line).argmin()
                     for start in (0, half) for line in LINE_V], dtype=float)


def _dip(x, p):
    return p[0] * (1 - p[1] * 0.25 / ((x - p[2]) ** 2 + 0.25))


def _flat(x, p):
    return np.full(len(x), p[0])


def _product(x, p):
    """A dip whose depth is p[1]*p[2]: the data see only the product."""
    return _dip(x, [p[0], p[1] * p[2], p[3]])


def _two(x, p):
    """Two spectra, each with its own baseline, depth and position."""
    return np.r_[_dip(x[:256], p[:3]), _dip(x[256:], p[3:])]


def _decay(t, p):
    """A time spectrum: 2000 counts falling to a few, with a quantum beat."""
    return p[0] * np.exp(-t / p[1]) * (1 + p[2] * np.cos(p[3] * t))


def _counts(model, x, p, seed):
    return np.random.default_rng(seed).poisson(model(x, p)).astype(float)


def test_a_fit_of_velocities_has_errors_and_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        p, errors, chi2, covariance, pearson = mi.minimi_hi(
            _drive, _line_channels(), np.tile(LINE_V, 2), np.array([-9.5, 0.28, 0.0]), tau0=1)
    assert np.all(np.isfinite(errors)) and np.all(errors > 0)
    assert np.isnan(pearson)                           # velocities are not counts
    # no Pearson's chi-square: nothing multiplies the covariance
    off = mi.minimi_hi(_drive, _line_channels(), np.tile(LINE_V, 2), np.array([-9.5, 0.28, 0.0]), tau0=1,
                       chi2_correction=False)
    assert errors == pytest.approx(off[1], rel=1e-12)


def test_spectra_that_share_no_parameter(capsys):
    y = _counts(_two, np.r_[X, X[::-1]], [1e5, 0.3, 0.2, 2e5, 0.2, -0.4], 1)
    p, errors, chi2, covariance, _ = mi.minimi_hi(_two, np.r_[X, X[::-1]], y,
                                                  np.array([1.01e5, 0.28, 0.25, 2.01e5, 0.22, -0.35]))
    assert 'inverse matrix' not in capsys.readouterr().out
    assert np.all(np.isfinite(errors)) and covariance.shape == (6, 6)


def test_a_parameter_the_others_reproduce_is_reported_like_a_fixed_one(capsys):
    y = _counts(_product, X, [1e5, 0.3, 0.5, 0.2], 2)

    p, errors, chi2, covariance, _ = mi.minimi_hi(_product, X, y, np.array([1.01e5, 0.25, 0.55, 0.1]))

    assert 'parameter No. 2 is not determined' in capsys.readouterr().out
    assert np.isnan(errors[2]) and np.all(np.isfinite(errors[[0, 1, 3]]))
    assert covariance.shape == (3, 3)
    # the same data with the depth as ONE parameter (one degree of freedom more)
    q, errors_q, _, _, _ = mi.minimi_hi(_dip, X, y, np.array([1.01e5, 0.25 * 0.55, 0.1]))
    assert p[1] * p[2] == pytest.approx(q[1], rel=1e-6)
    assert errors[[0, 3]] == pytest.approx(errors_q[[0, 2]], rel=5e-3)
    assert errors[1] * p[2] == pytest.approx(errors_q[1], rel=5e-3)


def test_a_bound_hit_after_the_last_jacobian_keeps_the_rows():
    y = _counts(_dip, X, [1e5, 0.3, 0.2], 3)
    bounds = np.array([[-np.inf, -np.inf, -np.inf], [np.inf, 0.2, np.inf]])

    p, errors, chi2, covariance, _ = mi.minimi_hi(_dip, X, y, np.array([1e5, 0.15, 0.1]),
                                                  bounds=bounds, MI=1, fixCH=1)

    assert p[1] == 0.2 and np.isnan(errors[1])
    assert covariance.shape == (2, 2)                  # p[0] and p[2], not p[1]
    assert errors[[0, 2]] == pytest.approx(np.sqrt(np.diag(covariance)), rel=1e-12)


@pytest.mark.parametrize('weights', ['U', 'P'])
def test_an_expression_of_one_parameter_has_its_error(weights):
    rng = np.random.default_rng(4)
    clean = _dip(X, [1e5, 0.3, 0.2])
    y = clean + 2 * np.sqrt(clean) * rng.normal(size=X.size)     # twice the Poisson noise

    p, errors, chi2, covariance, _ = mi.minimi_hi(_dip, X, y, np.array([1.01e5, 0.25, 0.1]),
                                                  weights=weights)

    assert chi2 > 3                                    # so a missing sqrt(chi2) would show
    # rel 1e-3: support_math differentiates with an absolute step of 1e-8,
    # a few hundred ulps of the 1e5 baseline
    for k in range(3):
        assert calculate_expression_error('p[%d]' % k, p, errors, covariance) \
            == pytest.approx(errors[k], rel=1e-3)


def test_the_chi_squares_returned_by_each_weighting():
    y = _counts(_dip, X, [1e3, 0.3, 0.2], 5)
    p0 = np.array([1.01e3, 0.25, 0.1])
    fits = {w: mi.minimi_hi(_dip, X, y, p0, weights=w) for w in ('U', 'W', 'P')}
    nu = len(X) - 3

    for w, (p, errors, chi2, covariance, pearson) in fits.items():
        m = _dip(X, p)
        assert pearson == pytest.approx(np.sum((y - m) ** 2 / m) / nu, rel=1e-12)
        if w == 'P':                                   # the deviance it minimised
            deviance = 2 * np.sum(m - y + np.where(y > 0, y * np.log(np.maximum(y, 1) / m), 0))
            assert chi2 == pytest.approx(deviance / nu, rel=1e-12)
        else:                                          # 'U' and 'W': the |y|+1 one
            assert chi2 == pytest.approx(np.sum((y - m) ** 2 / (np.abs(y) + 1)) / nu, rel=1e-12)
    assert fits['W'][2] <= fits['U'][2]                # 'W' minimises the one it reports


@pytest.mark.parametrize('weights', ['U', 'W', 'P'])
def test_the_chi2_correction_can_be_turned_off(weights):
    """Off, the errors are those of the variances alone: on, they are sqrt of
    Pearson's reduced chi-square (the last value returned) larger, whatever
    the weights."""
    rng = np.random.default_rng(11)
    clean = _dip(X, [1e5, 0.3, 0.2])
    y = clean + 2 * np.sqrt(clean) * rng.normal(size=X.size)     # twice the Poisson noise
    p0 = np.array([1.01e5, 0.25, 0.1])

    on = mi.minimi_hi(_dip, X, y, p0, weights=weights)
    off = mi.minimi_hi(_dip, X, y, p0, weights=weights, chi2_correction=False)

    assert off[0] == pytest.approx(on[0], rel=1e-12)              # the same fit
    factor = on[4]
    assert factor > 3
    assert on[1] == pytest.approx(off[1] * np.sqrt(factor), rel=1e-9)
    assert on[3] == pytest.approx(off[3] * factor, rel=1e-9)


def test_poisson_weights_maximise_the_likelihood():
    y = _counts(_dip, X, [1e3, 0.3, 0.2], 6)
    p0 = np.array([1.01e3, 0.25, 0.1])

    def loglik(p):
        m = _dip(X, p)
        return np.sum(y * np.log(m) - m)

    best = {w: loglik(mi.minimi_hi(_dip, X, y, p0, weights=w)[0]) for w in ('U', 'W', 'P')}
    assert best['P'] >= best['U'] and best['P'] >= best['W']


def test_an_array_of_data_weights_is_W():
    """The same fit, rounded differently: 'W' scales the rows by
    1/sqrt(|y|+1), the array by sqrt(1/(|y|+1)), an ulp apart. The two stop
    where the reduced chi-square drops by less than eps, so they agree to what
    that resolves, sigma*sqrt(eps*dof), not to rounding."""
    y = _counts(_dip, X, [1e3, 0.3, 0.2], 7)
    p0 = np.array([1.01e3, 0.25, 0.1])
    by_name = mi.minimi_hi(_dip, X, y, p0, weights='W')
    by_array = mi.minimi_hi(_dip, X, y, p0, weights=1 / (np.abs(y) + 1))
    eps = inspect.signature(mi.minimi_hi).parameters['eps'].default
    sigma = by_name[1] / np.sqrt(by_name[4])           # without Pearson's factor
    assert np.all(np.abs(by_array[0] - by_name[0]) < sigma * np.sqrt(eps * (len(X) - 3)))
    assert by_array[2] == pytest.approx(by_name[2], rel=1e-9)


def test_the_level_of_a_flat_spectrum():
    """512 points of 10 counts: Poisson and unweighted give the mean, data
    weights a level about one count lower (docs/minimization.md, 5.4)."""
    y = np.random.default_rng(8).poisson(10.0, 512).astype(float)
    level = {w: mi.minimi_hi(_flat, X[:1].repeat(512), y, np.array([9.0]), weights=w)[0][0]
             for w in ('U', 'W', 'P')}
    assert level['U'] == pytest.approx(y.mean(), rel=1e-9)
    assert level['P'] == pytest.approx(y.mean(), rel=1e-9)
    assert level['W'] < y.mean() - 0.5


def test_weights_that_cannot_be_used_are_refused():
    y = _counts(_dip, X, [1e3, 0.3, 0.2], 9)
    p0 = np.array([1.01e3, 0.25, 0.1])
    for weights in ('X', np.ones(10), -np.ones(len(X))):
        with pytest.raises(ValueError):
            mi.minimi_hi(_dip, X, y, p0, weights=weights)
    with pytest.raises(ValueError, match='counts'):    # velocities are not counts
        mi.minimi_hi(_drive, _line_channels(), np.tile(LINE_V, 2), np.array([-9.5, 0.28, 0.0]),
                     tau0=1, weights='P')


def test_poisson_weights_with_penalty_rows():
    """Recon-like: weights >= 0 behind smooth profiles, a smoothness penalty in
    trailing rows (n_reg) that keep the weight 1."""
    num, n = 7, 48
    t = np.linspace(0.0, 1.0, n)
    K = np.stack([np.exp(-0.5 * ((t - c / (num - 1)) / 0.16) ** 2) for c in range(num)], axis=1)
    data = np.random.default_rng(10).poisson(K @ np.array([0, 0.1, 0.5, 1.0, 0.5, 0.1, 0]) * 1e4 + 50)
    lam = 1e-4 * float(np.sum((data - data.mean()) ** 2 / (data + 1)))   # mild: the data decide

    def model(_x, w):
        g = w / max(np.sum(w), 1e-30)
        return np.concatenate([K @ w * 1e4 + 50, np.sqrt(lam) * (g[1:] - g[:-1])])

    w_fit, errors, chi2, covariance, pearson = mi.minimi_hi(
        model, np.zeros(n + num - 1), np.concatenate([data, np.zeros(num - 1)]), np.full(num, 0.3),
        bounds=np.array([[0.0] * num, [np.inf] * num]), n_reg=num - 1, MI=40, MI2=20, weights='P')

    assert np.all(np.isfinite(covariance)) and np.all(w_fit >= 0)
    assert 0.5 < chi2 < 2 and 0.5 < pearson < 2


def test_errors_match_the_scatter_of_repeated_fits():
    """400 fits of fresh Poisson data of a time spectrum: the reported errors
    against the scatter of the fitted values (itself uncertain by ~3.5 %). The
    formula before gave 0.6 for the decay time and the beat frequency. The
    Poisson fit is the more precise one here: its scatter is 0.6 of the
    unweighted fit's for those two."""
    rng = np.random.default_rng(2026)
    t = np.arange(200, dtype=float)
    truth = np.array([2000.0, 30.0, 0.6, 0.25])
    fits = {'U': [], 'P': []}
    errors = {'U': [], 'P': []}
    for _ in range(400):
        y = rng.poisson(_decay(t, truth)).astype(float)
        start = truth * (1 + 0.01 * rng.normal(size=4))
        for w in fits:
            p, e, _, _, _ = mi.minimi_hi(_decay, t, y, start, weights=w)
            fits[w].append(p)
            errors[w].append(e)
    scatter = {w: np.std(fits[w], axis=0, ddof=1) for w in fits}
    for w in fits:
        assert np.median(errors[w], axis=0) / scatter[w] == pytest.approx(np.ones(4), abs=0.25)
    assert np.all(scatter['P'][[1, 3]] < 0.8 * scatter['U'][[1, 3]])
