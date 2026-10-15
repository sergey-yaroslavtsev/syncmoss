"""Cosine smearing of CMS spectra (models.Voight_box / Voight_disc, TI's
``dtheta``, ``v_center`` and ``r_ratio``), its storage next to G (#@GCMS and
GCMS.txt: G, Δθ, then r/R), the searches that fit G or Δθ, and the error function
it is built on (models.erf, Abramowitz & Stegun 7.1.26).

A photon leaving the moving source at the angle theta to the drive axis sees
the drive velocity times cos(theta). For a point source cos(theta) is uniform on
[cos(dtheta), 1]; the velocity calibration already holds the mean factor, and
the drive is at rest at v_center, the middle of the velocity range. So a ray
sees v_center + (v - v_center)*s, s = cos(theta)/<cos>. TI averages the SOURCE
line over the matching box of width MulCo*(v - v_center)*(1 - cos dtheta)/<cos>
at the energies the absorber is computed at anyway (one absorber evaluation per
node, as without smearing).

Pinned: erf on arrays; the cumulative and the box average of Voight; dtheta = 0
is the former model and SMS ignores dtheta; TI against the exact smeared
integral (absorber-energy integration averaged over s), about 0 and about a
shifted centre; a part of the points gives the same values; areas and
positions kept, broadening growing as (v - v_center)^2; Nbaseline sections, the
node-sum store and TI_recon_variants carrying both; the callers that pass them
on; the #@GCMS / GCMS.txt second number; the GUI field next to G. Serial pool.
"""
import json
import math
import os

import numpy as np
import pytest

import syncmoss.models as m5
from syncmoss import instrumental_io as iio
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import (NAT_WIDTH, number_of_baseline_parameters as NB,
                                ALPHA_FE_V_OUTER, LINE_RATIO_25, LINE_RATIO_34)

from conftest import FROZEN_PARAMETERS


class _SerialPool:
    def starmap(self, func, iterable):
        return [func(*a) for a in iterable]


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_G = float(_frozen("GCMS.txt", delimiter='\t'))
_MULCO_SMS, _X0_SMS = (float(v) for v in _frozen("INSint.txt"))
_INS_SMS = np.atleast_1d(_frozen("INSexp.txt"))
_DTHETA = 0.2                                     # rad; the MossLab spectra need ~0.195


@pytest.fixture(autouse=True)
def _empty_store():
    m5._NODE_SUMS.clear()
    yield
    m5._NODE_SUMS.clear()


def _cms(x, p, model, JN, dtheta=0.0, Norm=1.0, lengths=None, Recon=[0], v_center=0.0, r_ratio=0.0):
    return np.asarray(m5.TI(np.asarray(x, dtype=float), np.asarray(p, dtype=float), list(model), JN,
                            _SerialPool(), 0.0, MulCoCMS, _G, [0], [0], Met=1, Norm=Norm,
                            lengths=lengths, Recon=Recon, dtheta=dtheta, v_center=v_center,
                            r_ratio=r_ratio), dtype=float)


def _singlets(lines, N0=1.0):
    """Baseline (N0 only) + natural-width singlets [(position, thickness), ...]."""
    p = np.zeros(NB)
    p[0] = N0
    return np.concatenate([p] + [[t, x, NAT_WIDTH, 0.0] for x, t in lines])


def _scales(dtheta, n):
    """Gauss-Legendre nodes and weights (summing to 1) of the velocity scale
    s = cos(theta)/<cos>, cos(theta) uniform on [cos(dtheta), 1]."""
    c = np.cos(dtheta)
    lo, hi = c / ((1 + c) / 2), 1 / ((1 + c) / 2)
    t, w = np.polynomial.legendre.leggauss(n)
    return (lo + hi) / 2 + (hi - lo) / 2 * t, w / 2


# --- erf --------------------------------------------------------------------------

def test_erf_takes_arrays_element_by_element():
    x = np.concatenate([np.linspace(-6.0, 6.0, 12001), [0.0, -0.0, 1e-300, 30.0, -30.0]])
    got = m5.erf(x)
    one_by_one = np.array([m5.erf(v) for v in x])
    np.testing.assert_allclose(got, one_by_one, rtol=0, atol=4 * np.finfo(float).eps)
    assert np.abs(got - np.array([math.erf(v) for v in x])).max() < 1.5e-7     # A&S 7.1.26
    assert m5.erf(np.ones((3, 4))).shape == (3, 4)


def test_limits_keeps_its_erf():
    """limits() steers the SMS instrumental-function searches, which must give
    the numbers they gave: a scalar goes through the formula as before."""
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    t = 1.0 / (1.0 + p * 0.5)
    assert m5.erf(0.5) == 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-0.5 * 0.5)


# --- the cumulative and the box average of Voight ---------------------------------

_GL, _GG = NAT_WIDTH * MulCoCMS, _G * MulCoCMS
_U = np.linspace(-3.0, 3.0, 6001)
_PEAK = m5.Voight(_GL, _GG, np.array([0.0]))[0]
# erf is good to 1.5e-7; the slope of its cumulative -- what a narrow box sees --
# to ~1e-5 of the peak (9e-6 at this G), a box of 1e-3 and wider to 1.5e-6
_SLOPE_TOL, _BOX_TOL = 2e-5, 5e-6


def test_voight_cdf_is_the_integral_of_voight():
    h = 1e-6
    slope = (m5._voight_cdf(_GL, _GG, _U + h) - m5._voight_cdf(_GL, _GG, _U - h)) / (2 * h)
    assert np.abs(slope - m5.Voight(_GL, _GG, _U)).max() < _SLOPE_TOL * _PEAK
    ends = m5._voight_cdf(_GL, _GG, np.array([-1e8, 1e8]))
    assert ends[0] == pytest.approx(0.0, abs=1e-6) and ends[1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("W", [1e-3, 0.02, 0.06, -0.06])
def test_voight_box_is_voight_averaged_over_the_box(W):
    u = _U[::10]
    t = np.linspace(-abs(W) / 2, abs(W) / 2, 2001)
    brute = np.trapezoid(m5.Voight(_GL, _GG, (u[:, None] + t[None, :]).ravel()).reshape(len(u), len(t)),
                         t, axis=1) / abs(W)
    got = m5.Voight_box(_GL, _GG, u, np.full_like(u, W))
    assert np.abs(got - brute).max() < _BOX_TOL * _PEAK


def test_voight_box_too_narrow_to_matter_is_the_point_value():
    width = _GL + _GG
    limit = m5._BOX_POINT_LIMIT * width
    point = m5.Voight(_GL, _GG, _U)
    assert np.array_equal(m5.Voight_box(_GL, _GG, _U, np.full_like(_U, limit / 2)), point)
    assert np.array_equal(m5.Voight_box(_GL, _GG, _U, np.zeros_like(_U)), point)
    # just above the switch the box average is the point value within erf's slope
    above = m5.Voight_box(_GL, _GG, _U, np.full_like(_U, 2 * limit))
    assert np.abs(above - point).max() < 3 * _SLOPE_TOL * _PEAK


# --- the disc source: Voight averaged over the disc kernel ------------------------

def _lens(d, r1, r2):
    """Overlap area of two discs of radii r1 and r2, centres d apart."""
    d = np.asarray(d, dtype=float)
    area = np.where(d <= abs(r1 - r2), np.pi * min(r1, r2) ** 2, 0.0)
    mid = (d > abs(r1 - r2)) & (d < r1 + r2)
    dm = d[mid]
    area[mid] = (r1 ** 2 * np.arccos((dm ** 2 + r1 ** 2 - r2 ** 2) / (2 * dm * r1))
                 + r2 ** 2 * np.arccos((dm ** 2 + r2 ** 2 - r1 ** 2) / (2 * dm * r2))
                 - 0.5 * np.sqrt((-dm + r1 + r2) * (dm + r1 - r2) * (dm - r1 + r2) * (dm + r1 + r2)))
    return area


def _disc_density(x, q):
    """Density of q = (rho/R)^2 over the rays of a disc source of radius x*R
    and an aperture of radius R: rho, the offset of a ray in the aperture plane
    from where it left the source, is the difference of a point of each disc, so
    its density in the plane is their overlap at distance rho -- and dq is
    rho drho (times 2), so in q it is that overlap itself."""
    return _lens(np.sqrt(q), 1.0, x)


@pytest.mark.parametrize("x", [0.3, 1.0, 2.5])
@pytest.mark.parametrize("W", [0.02, -0.06, 0.15])
def test_voight_disc_is_voight_averaged_over_the_disc_kernel(x, W):
    """The kernel is the ray density in q scaled to the spread of the point box
    of width W (tau = (q - <q>)/(sqrt(12) std q)); a fine sum over q against
    the few nodes of the Gauss rule."""
    q = np.linspace(0.0, (1 + x) ** 2, 40001)
    w = _disc_density(x, q)
    w[[0, -1]] /= 2
    w /= w.sum()
    mean = np.sum(w * q)
    tau = (q - mean) / np.sqrt(12 * np.sum(w * (q - mean) ** 2))
    u = _U[::10]
    keep = np.flatnonzero(w)[::8]
    brute = sum(wi * m5.Voight(_GL, _GG, u + W * t) for t, wi in zip(tau[keep], w[keep])) / w[keep].sum()
    got = m5.Voight_disc(_GL, _GG, u, np.full_like(u, W), x)
    assert np.abs(got - brute).max() < 1e-5 * _PEAK


def test_disc_rule_has_the_spread_of_the_point_box():
    """Weights summing to 1, mean 0 and the variance of the unit box (1/12),
    whatever r/R; r and R play the same part."""
    for x in (1e-3, 0.3, 1.0, 2.5):
        for nodes in (m5.DISC_NODES_MIN, 2 * m5.DISC_NODES_MIN):
            tau, w = m5.disc_rule(x, nodes)
            assert len(tau) == nodes
            assert w.sum() == pytest.approx(1.0, abs=1e-14) and np.sum(w * tau) == pytest.approx(0.0, abs=1e-14)
            assert np.sum(w * tau ** 2) == pytest.approx(1 / 12, rel=1e-12)
    for a, b in zip(m5.disc_rule(0.6), m5.disc_rule(1 / 0.6)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)


def test_a_small_disc_is_the_point_source():
    """r/R -> 0: the rays of a point source, cos(theta) uniform -- the box."""
    for W in (0.02, 0.06, 0.15):
        box = m5.Voight_box(_GL, _GG, _U, np.full_like(_U, W))
        assert np.abs(m5.Voight_disc(_GL, _GG, _U, np.full_like(_U, W), 1e-4) - box).max() < 1e-5 * _PEAK


# --- TI: off, SMS, and the exact smeared integral ---------------------------------

_FE = ALPHA_FE_V_OUTER * np.array([-1, -LINE_RATIO_25, -LINE_RATIO_34, LINE_RATIO_34, LINE_RATIO_25, 1])
_ALPHA_FE = {"thin": list(zip(_FE, 1.0 * np.array([3, 2, 1, 1, 2, 3]) / 12)),
             "thick": list(zip(_FE, 15.0 * np.array([3, 2, 1, 1, 2, 3]) / 12))}


def test_zero_dtheta_is_the_former_model():
    x = np.linspace(-8.0, 8.0, 41)
    p = _singlets(_ALPHA_FE["thick"], 2.0)
    former = np.asarray(m5.TI(x, p, ['Singlet'] * 6, 32, _SerialPool(), 0.0, MulCoCMS, _G, [0], [0],
                              Met=1, Norm=0.99), dtype=float)
    assert np.array_equal(_cms(x, p, ['Singlet'] * 6, 32, dtheta=0.0, Norm=0.99, v_center=0.37), former)
    assert np.array_equal(_cms(x, p, ['Singlet'] * 6, 32, dtheta=0.0, Norm=0.99, v_center=0.37, r_ratio=0.5),
                          former)


def test_sms_ignores_dtheta():
    x = np.linspace(-6.0, 6.0, 41)
    p = _singlets([(3.0, 2.0)])

    def sms(**kw):
        return np.asarray(m5.TI(x, p, ['Singlet'], 16, _SerialPool(), _X0_SMS, _MULCO_SMS, _INS_SMS,
                                [0], [0], Met=0, **kw), dtype=float)
    assert np.array_equal(sms(dtheta=_DTHETA, v_center=0.3, r_ratio=0.5), sms())


def _sigma(E, lines):
    return sum(np.pi * NAT_WIDTH / 2 * t * m5.Voight(NAT_WIDTH, 0.0, E - x) for x, t in lines)


def _exact(v, lines):
    """The N0 = 1 CMS count rate integrated over the ABSORBER energy (no window,
    no Norm; see test_source_norm._exact_cms)."""
    h = NAT_WIDTH / 20
    pos = [x for x, _ in lines]
    E = np.arange(min(pos) - 40.0, max(pos) + 40.0 + h / 2, h)
    absorbed = 1 - np.exp(-_sigma(E, lines))
    source = m5.Voight(NAT_WIDTH, _G, (E[None, :] - np.asarray(v)[:, None]).ravel()).reshape(len(v), len(E))
    return 1 - source @ absorbed * h


def _exact_smeared(v, lines, dtheta, v_center=0.0):
    """_exact averaged over the rays: velocity v_center + (v - v_center)*s
    (Gauss-Legendre in s: the spectrum is smooth over the 2 % of scale it spans)."""
    s, w = _scales(dtheta, 8)
    return sum(wi * _exact(v_center + (v - v_center) * si, lines) for si, wi in zip(s, w))


@pytest.mark.parametrize("name", sorted(_ALPHA_FE))
def test_smeared_model_follows_the_exact_smeared_integral(name):
    """With dtheta, TI is as close to the exact smeared spectrum as it is to the
    exact one without: the smearing adds no error of its own. And what is
    compared is not small: the smearing moves the spectrum far more."""
    JN = 64
    lines = _ALPHA_FE[name]
    v = np.linspace(-8.0, 8.0, 161)                     # 0.1 mm/s: on the line cores
    p = _singlets(lines)
    Norm = iio.compute_norm(_SerialPool(), JN, dict(x0=0.0, MulCo=MulCoCMS, INS=_G, Met=1))
    plain_error = np.abs(_cms(v, p, ['Singlet'] * 6, JN, 0.0, Norm) - _exact(v, lines)).max()
    exact_smeared = _exact_smeared(v, lines, _DTHETA)
    smeared_error = np.abs(_cms(v, p, ['Singlet'] * 6, JN, _DTHETA, Norm) - exact_smeared).max()
    assert smeared_error < 1.25 * plain_error
    assert np.abs(exact_smeared - _exact(v, lines)).max() > 10 * smeared_error


def test_smearing_starts_where_the_drive_is_at_rest():
    """About a centre that is not 0 the smearing follows the exact integral as
    well, and differs from the smearing about 0 by more than that error."""
    JN = 64
    lines = _ALPHA_FE["thick"]
    v = np.linspace(-7.6, 8.4, 161)                     # the drive at rest at +0.4 mm/s
    v_center = iio.drive_rest_velocity(v)
    assert v_center == pytest.approx(0.4)
    p = _singlets(lines)
    Norm = iio.compute_norm(_SerialPool(), JN, dict(x0=0.0, MulCo=MulCoCMS, INS=_G, Met=1))
    plain_error = np.abs(_cms(v, p, ['Singlet'] * 6, JN, 0.0, Norm) - _exact(v, lines)).max()
    model = _cms(v, p, ['Singlet'] * 6, JN, _DTHETA, Norm, v_center=v_center)
    error = np.abs(model - _exact_smeared(v, lines, _DTHETA, v_center)).max()
    assert error < 1.25 * plain_error
    assert np.abs(model - _cms(v, p, ['Singlet'] * 6, JN, _DTHETA, Norm)).max() > 2 * error


def _exact_disc(v, lines, x, dtheta, v_center=0.0):
    """_exact averaged over the rays of a disc source of r/R = x: q from its own
    density (_disc_density, Gauss-Legendre on the flat part and on the lens),
    the velocity scale s = 1 - k*tau, k = (1 - cos dtheta)/<cos> the width of
    the point source's box with the same spread (1 - cos(theta) ~ q, as the
    model takes it)."""
    pieces = [(0.0, (1 - x) ** 2, 8), ((1 - x) ** 2, (1 + x) ** 2, 24)]
    q, w = [], []
    for a, b, n in pieces:
        t, wt = np.polynomial.legendre.leggauss(n)
        q.append((a + b) / 2 + (b - a) / 2 * t)
        w.append((b - a) / 2 * wt)
    q, w = np.concatenate(q), np.concatenate(w)
    w = w * _disc_density(x, q)
    w /= w.sum()
    mean = np.sum(w * q)
    tau = (q - mean) / np.sqrt(12 * np.sum(w * (q - mean) ** 2))
    k = (1 - np.cos(dtheta)) / ((1 + np.cos(dtheta)) / 2)
    return sum(wi * _exact(v_center + (v - v_center) * (1 - k * ti), lines) for ti, wi in zip(tau, w))


def test_disc_model_follows_the_exact_disc_integral():
    """With a disc source TI is as close to the exact spectrum as without
    smearing; the point source with the same Δθ is not."""
    JN, x, v_center = 64, 0.5, 0.4
    lines = _ALPHA_FE["thick"]
    v = np.linspace(-7.6, 8.4, 81)
    p = _singlets(lines)
    Norm = iio.compute_norm(_SerialPool(), JN, dict(x0=0.0, MulCo=MulCoCMS, INS=_G, Met=1))
    plain_error = np.abs(_cms(v, p, ['Singlet'] * 6, JN, 0.0, Norm) - _exact(v, lines)).max()
    exact_disc = _exact_disc(v, lines, x, _DTHETA, v_center)
    disc = _cms(v, p, ['Singlet'] * 6, JN, _DTHETA, Norm, v_center=v_center, r_ratio=x)
    point = _cms(v, p, ['Singlet'] * 6, JN, _DTHETA, Norm, v_center=v_center)
    error = np.abs(disc - exact_disc).max()
    assert error < 1.25 * plain_error
    assert np.abs(point - exact_disc).max() > 2 * error


def test_a_part_of_the_points_gives_the_same_values():
    """v_center comes with the call, not from the points: a fit on the points left
    by exclusion regions sees the model of the whole spectrum."""
    v = np.linspace(-10.0, 10.8, 105)
    p = _singlets(_ALPHA_FE["thin"])
    whole = _cms(v, p, ['Singlet'] * 6, 32, _DTHETA, v_center=0.4)
    keep = (v > -9.0) & (v < 2.0)                        # one edge and a side cut away
    np.testing.assert_allclose(_cms(v[keep], p, ['Singlet'] * 6, 32, _DTHETA, v_center=0.4), whole[keep],
                               rtol=1e-13, atol=0)


def test_smearing_keeps_areas_and_positions_and_grows_as_v_squared():
    """Line areas stay (to the variance of s, 3e-5) and so do centroids; the
    dip of a line at 6 mm/s gets shallower by several %, of a line at 0.84
    mm/s by (0.84/5.31)^2 of what one at 5.31 loses, of a line at 0 by nothing."""
    JN = 64
    v = np.linspace(-12.0, 12.0, 2401)

    def dip(x, dtheta):
        return 1 - _cms(v, _singlets([(x, 0.3)]), ['Singlet'], JN, dtheta)

    plain, smeared = dip(6.0, 0.0), dip(6.0, _DTHETA)
    area0, area = np.trapezoid(plain, v), np.trapezoid(smeared, v)
    assert abs(area / area0 - 1) < 1e-4
    assert abs(np.trapezoid(smeared * v, v) / area - np.trapezoid(plain * v, v) / area0) < 1e-3
    assert smeared.max() < 0.96 * plain.max()

    loss = {x: dip(x, 0.0).max() - dip(x, _DTHETA).max() for x in (0.0, LINE_RATIO_34 * ALPHA_FE_V_OUTER,
                                                                  ALPHA_FE_V_OUTER)}
    inner, outer = LINE_RATIO_34 * ALPHA_FE_V_OUTER, ALPHA_FE_V_OUTER
    assert loss[inner] / loss[outer] == pytest.approx((inner / outer) ** 2, rel=0.15)
    assert abs(loss[0.0]) < 1e-3 * loss[outer]


# --- Nbaseline sections, the node-sum store, TI_recon_variants --------------------

def test_a_cms_section_is_smeared_and_an_sms_section_is_not():
    x = np.linspace(-8.0, 8.0, 41)
    n = len(x)
    p_cms, p_sms = _singlets([(5.0, 3.0)], 2.0), _singlets([(5.0, 1.0)], 3.0)
    for dtheta, v_center in ((_DTHETA, 0.3), ([_DTHETA, _DTHETA], [0.3, -2.0]), ([_DTHETA, 0.0], [0.3, 0.0])):
        mixed = np.asarray(m5.TI(np.concatenate([x, x]), np.concatenate([p_cms, p_sms]),
                                 ['Singlet', 'Nbaseline', 'Singlet'], 16, _SerialPool(),
                                 [0.0, _X0_SMS], [MulCoCMS, _MULCO_SMS], [_G, _INS_SMS], [0], [0],
                                 Met=[1, 0], Norm=[1, 1], lengths=[n, n], dtheta=dtheta,
                                 v_center=v_center), dtype=float)
        np.testing.assert_allclose(mixed[:n], _cms(x, p_cms, ['Singlet'], 16, _DTHETA, v_center=0.3),
                                   rtol=1e-12, atol=0)
        sms = np.asarray(m5.TI(x, p_sms, ['Singlet'], 16, _SerialPool(), _X0_SMS, _MULCO_SMS, _INS_SMS,
                               [0], [0], Met=0), dtype=float)
        np.testing.assert_allclose(mixed[n:], sms, rtol=1e-12, atol=0)
    assert not np.allclose(_cms(x, p_cms, ['Singlet'], 16, _DTHETA), _cms(x, p_cms, ['Singlet'], 16, 0.0))


def test_each_section_has_its_own_r_over_r():
    x = np.linspace(-8.0, 8.0, 41)
    n = len(x)
    p = np.concatenate([_singlets([(5.0, 3.0)]), _singlets([(-5.0, 2.0)])])
    both = _cms(np.concatenate([x, x]), p, ['Singlet', 'Nbaseline', 'Singlet'], 16, _DTHETA, lengths=[n, n],
                v_center=[0.3, 0.3], r_ratio=[0.5, 0.0])
    disc = _cms(x, p[:NB + 4], ['Singlet'], 16, _DTHETA, v_center=0.3, r_ratio=0.5)
    point = _cms(x, p[NB + 4:], ['Singlet'], 16, _DTHETA, v_center=0.3)
    np.testing.assert_allclose(both[:n], disc, rtol=1e-12, atol=0)
    np.testing.assert_allclose(both[n:], point, rtol=1e-12, atol=0)
    assert not np.allclose(disc, _cms(x, p[:NB + 4], ['Singlet'], 16, _DTHETA, v_center=0.3), rtol=1e-6, atol=0)


def test_the_node_sum_store_tells_dtheta_v_center_and_r_over_r_apart():
    x = np.linspace(-8.0, 8.0, 41)
    n = len(x)
    p = np.concatenate([_singlets([(5.0, 3.0)]), _singlets([(-5.0, 2.0)])])
    model = ['Singlet', 'Nbaseline', 'Singlet']
    xx = np.concatenate([x, x])
    off = _cms(xx, p, model, 16, 0.0, lengths=[n, n])
    on = _cms(xx, p, model, 16, _DTHETA, lengths=[n, n])
    moved = _cms(xx, p, model, 16, _DTHETA, lengths=[n, n], v_center=[0.0, 1.0])
    disc = _cms(xx, p, model, 16, _DTHETA, lengths=[n, n], r_ratio=[0.0, 0.5])
    assert not np.allclose(on, off) and not np.allclose(moved[n:], on[n:])
    assert not np.allclose(disc[n:], on[n:], rtol=1e-6, atol=0)
    assert np.array_equal(moved[:n], on[:n]) and np.array_equal(disc[:n], on[:n])
    assert np.array_equal(_cms(xx, p, model, 16, 0.0, lengths=[n, n]), off)
    m5._NODE_SUMS.clear()
    assert np.array_equal(_cms(xx, p, model, 16, _DTHETA, lengths=[n, n]), on)
    m5._NODE_SUMS.clear()
    assert np.array_equal(_cms(xx, p, model, 16, _DTHETA, lengths=[n, n], r_ratio=[0.0, 0.5]), disc)


with open(os.path.join(os.path.dirname(__file__), "golden_models.json")) as _f:
    _SEXTET = list(json.load(_f)["cases"]["Sextet"]["params"])
_SEXTET[0] = 10.0                                         # thick
_RECON_H = [3.0, _SEXTET[3] - 3.0, _SEXTET[3] + 3.0, 5.0, 0.0, 0.0, 0.0]
_BASE = [1.0e4] + [0.0] * (NB - 1)
_W = [np.array([0.1, 0.3, 0.4, 0.15, 0.05]), np.array([0.2, 0.2, 0.2, 0.2, 0.2])]


@pytest.mark.parametrize("sections", [1, 2])
def test_recon_variants_carry_dtheta_v_center_and_r_over_r(sections):
    """One spectrum (its nodes in blocks for all the weight lists) and two (the
    first shared, the second in blocks): every list as TI gives it."""
    x1 = np.linspace(-10.0, 10.0, 41)
    if sections == 1:
        model, p, lengths, x, vc, rr = ['Sextet', 'Recon'], _BASE + _SEXTET + _RECON_H, None, x1, 0.3, 0.5
    else:
        model = ['Sextet', 'Nbaseline', 'Sextet', 'Recon']
        p, lengths, x = _BASE + _SEXTET + _BASE + _SEXTET + _RECON_H, [len(x1)] * 2, np.concatenate([x1, x1])
        vc, rr = [0.3, -0.2], [0.5, 1.0]
    p = np.array(p, dtype=float)
    recons = [[w] for w in _W]
    got = m5.TI_recon_variants(x, p, model, 16, _SerialPool(), 0.0, MulCoCMS, _G, [0], [0], 1, 1,
                               m5.SMS_POL_DEFAULT, recons, lengths, dtheta=_DTHETA, v_center=vc, r_ratio=rr)
    for R, spectrum in zip(recons, got):
        m5._NODE_SUMS.clear()
        np.testing.assert_allclose(np.asarray(spectrum, dtype=float),
                                   _cms(x, p, model, 16, _DTHETA, lengths=lengths, Recon=R, v_center=vc,
                                        r_ratio=rr),
                                   rtol=1e-12, atol=0)
        assert not np.allclose(np.asarray(spectrum, dtype=float),
                               _cms(x, p, model, 16, _DTHETA, lengths=lengths, Recon=R, v_center=vc),
                               rtol=1e-6, atol=0)


# --- the callers pass the resolved method's dtheta, v_center and r_ratio on -------

def _mp(dtheta, v_center, r_ratio=0.0):
    mp = dict(method='CMS', Met=1, INS=_G, x0=0.0, MulCo=MulCoCMS, dtheta=dtheta, v_center=v_center,
              r_ratio=r_ratio)
    mp['Norm'] = iio.compute_norm(_SerialPool(), 16, mp)
    return mp


@pytest.mark.parametrize("r_ratio", [0.0, 0.5])
def test_component_curves_hires_check_and_model_spectrum_use_the_smearing(r_ratio):
    from syncmoss import fitting_io
    from syncmoss.spectrum_io import compute_model_spectrum
    x = np.linspace(-8.0, 8.0, 41)
    p = _singlets([(5.0, 3.0)])
    mp = _mp(_DTHETA, 0.3, r_ratio)
    smeared = _cms(x, p, ['Singlet'], 16, _DTHETA, mp['Norm'], v_center=0.3, r_ratio=r_ratio)
    FS, _ = fitting_io.compute_component_curves(x, [p], [['Singlet']], [], [], 16, _SerialPool(), mp,
                                                m5.SMS_POL_DEFAULT)
    np.testing.assert_allclose(np.asarray(FS[0], dtype=float), smeared, rtol=1e-12, atol=0)
    hires = np.asarray(iio.hires_model_diff(_SerialPool(), 16, x, p, ['Singlet'], mp, smeared), dtype=float)
    hi = _cms(x, p, ['Singlet'], 16 * iio.HIRES_INTEGRATION_FACTOR, _DTHETA,
              iio.compute_norm(_SerialPool(), 16 * iio.HIRES_INTEGRATION_FACTOR, mp), v_center=0.3,
              r_ratio=r_ratio)
    np.testing.assert_allclose(hires, hi - smeared, rtol=0, atol=1e-12)
    job = dict(A=x, p=p, model=['Singlet'], JN=16, method_params=mp, Distri=[0], Cor=[0], Recon=[0],
               pol=m5.SMS_POL_DEFAULT)
    np.testing.assert_allclose(compute_model_spectrum(job, _SerialPool()), smeared, rtol=1e-12, atol=0)


def test_same_method_params_tells_dtheta_and_r_over_r_apart():
    a = dict(method='CMS', Met=1, INS=_G, x0=0.0, MulCo=MulCoCMS, dtheta=_DTHETA, r_ratio=0.5)
    assert iio.same_method_params(a, dict(a))
    assert not iio.same_method_params(a, dict(a, dtheta=0.0))
    assert not iio.same_method_params(a, dict(a, r_ratio=0.0))
    plain = dict(a, dtheta=0.0, r_ratio=0.0)
    assert iio.same_method_params(plain, {k: v for k, v in a.items() if k not in ('dtheta', 'r_ratio')})


# --- #@GCMS and GCMS.txt: G, then Δθ (none in files written before: 0) -------------

def _write_dat(directory, name, source_dat, header_lines):
    dst = os.path.join(str(directory), name)
    with open(source_dat) as src, open(dst, "w") as out:
        out.write("# Converted by SYNCMoss\n")
        for line in header_lines:
            out.write(line + "\n")
        for line in src:
            if line.strip() and not line.lstrip().startswith(("#", "<")):
                out.write(line)
    return dst


def test_gcms_header_numbers_are_g_the_smearing_and_the_diameters(tmp_path):
    """#@GCMS G Δθ source aperture (mm); G alone, as written before: no
    smearing (the diameters then do not matter)."""
    cal = os.path.join(FROZEN_PARAMETERS, "Calibration.dat")
    disc = _write_dat(tmp_path, "disc.dat", cal, ["#@GCMS 0.123 0.195 3.0 5.0"])
    same_ratio = _write_dat(tmp_path, "same_ratio.dat", cal, ["#@GCMS 0.123 0.195 6.0 10.0"])
    old = _write_dat(tmp_path, "old.dat", cal, ["#@GCMS 0.123"])
    meta = [iio.parse_dat_instrumental_metadata(f) for f in (disc, same_ratio, old)]
    assert [(m['GCMS'], m['dtheta'], m['source_diameter'], m['aperture_diameter'], m['r_ratio']) for m in meta] \
        == [(0.123, 0.195, 3.0, 5.0, 0.6), (0.123, 0.195, 6.0, 10.0, 0.6), (0.123, 0.0, 5.0, 20.0, 0.25)]
    assert len({iio.dat_metadata_key(f) for f in (disc, same_ratio, old)}) == 3
    assert iio.shared_dat_metadata_lines([disc, old]) == []          # not the same instrument
    resolved = iio.resolve_instrumental_for_file(None, disc, force_method='CMS')
    assert (resolved['dtheta'], resolved['r_ratio']) == (0.195, 0.6) and "d/D = 0.6" in resolved['note']


def test_gcms_file_keeps_g_the_smearing_and_the_diameters(tmp_path):
    """G, Δθ, then the source and aperture diameters d and D (mm); G alone, as
    written before: no smearing, the default 5 mm source and 20 mm aperture
    (d/D 0.25, the least bad guess)."""
    defaults = (iio.DEFAULT_SOURCE_DIAMETER, iio.DEFAULT_APERTURE_DIAMETER)
    assert defaults == (5.0, 20.0)
    (tmp_path / "GCMS.txt").write_text("0.097")                      # as written before
    assert iio.read_gcms_file(str(tmp_path)) == (0.097, 0.0) + defaults
    iio.write_gcms_file(str(tmp_path), 0.0764912345, 0.19532167, 1.25, 7.5)
    assert iio.read_gcms_file(str(tmp_path)) == (0.0764912345, 0.19532167, 1.25, 7.5)   # every digit


# --- the window ------------------------------------------------------------------

@pytest.mark.gui
def test_dtheta_field_takes_the_aps_place_next_to_g(physics_app):
    from PySide6.QtGui import QValidator
    from PySide6.QtWidgets import QCheckBox
    assert not hasattr(physics_app, 'APS_fit')
    assert all(box.text() != "APS" for box in physics_app.findChildren(QCheckBox))
    grid = physics_app.GCMS_input.parentWidget().layout()

    def cell(widget):
        """(row, column) of the grid cell holding *widget*, and that cell's label."""
        for i in range(grid.count()):
            inner = grid.itemAt(i).layout()
            if inner is not None and inner.indexOf(widget) >= 0:
                return grid.getItemPosition(i)[:2], inner.itemAt(0).widget()
    (g_cell, g_label), (d_cell, d_label) = cell(physics_app.GCMS_input), cell(physics_app.dtheta_input)
    assert (g_cell, g_label.text()) == ((1, 0), "G") and (d_cell, d_label.text()) == ((1, 1), "Δθ")
    assert grid.columnCount() == 2                                   # r/R is not here: a Supp setting
    assert not hasattr(physics_app, 'r_ratio_input')
    assert physics_app.source_diameter_input.isHidden() and physics_app.aperture_diameter_input.isHidden()
    assert physics_app.dtheta_input.text() == "0"                    # the frozen GCMS.txt holds G alone
    for widget in (physics_app.GCMS_input, physics_app.dtheta_input, g_label, d_label):
        assert "CMS only" in widget.toolTip()
    validator = physics_app.dtheta_input.validator()
    assert validator.validate("0.195", 0)[0] == QValidator.State.Acceptable
    assert validator.validate("-0.1", 0)[0] != QValidator.State.Acceptable


@pytest.mark.gui
def test_cms_source_fields_show_four_decimals_and_keep_every_digit(physics_app):
    """Like a parameter's value field: the full value is what is read and
    written, CMS_SOURCE_DECIMALS of it are shown, the rest in the tooltip."""
    from PySide6.QtWidgets import QLineEdit
    physics_app.GCMS_input.setText("0.0764912345")
    physics_app.dtheta_input.setText("0.19532167")
    try:
        assert QLineEdit.text(physics_app.GCMS_input) == "0.0765"
        assert QLineEdit.text(physics_app.dtheta_input) == "0.1953"
        assert iio.get_internal_gcms(physics_app) == 0.0764912345
        assert iio.get_internal_dtheta(physics_app) == 0.19532167
        tip = physics_app.dtheta_input.toolTip()
        assert "CMS only" in tip and "0.19532167" in tip
        physics_app.MS_fit.setChecked(True)
        # with the default diameters: a 5 mm source, a 20 mm aperture
        assert iio.build_dat_metadata_lines(physics_app)[0] == ["#@GCMS 0.0764912345 0.19532167 5.0 20.0"]
    finally:
        physics_app.SMS_fit.setChecked(True)
        physics_app.GCMS_input.setText(str(_G))
        physics_app.dtheta_input.setText("0")
    assert "CMS only" in physics_app.dtheta_input.toolTip()


@pytest.mark.gui
def test_save_g_and_dtheta_as_defaults(physics_app):
    """Instrumental function -> Save G, Δθ, d and D as defaults: every digit of
    G and Δθ (into the test's own copy of parameters/), with the diameters."""
    texts = [a.text() for a in physics_app.instrumental_menu.actions()]
    assert "Save G, Δθ, d and D as defaults" in texts
    assert "Reset instrumental function to default values" in texts
    physics_app.GCMS_input.setText("0.0764912345")
    physics_app.dtheta_input.setText("0.19532167")
    physics_app.source_diameter_input.setText("2.5")
    try:
        physics_app.save_gcms_pressed()
        assert iio.read_gcms_file(physics_app.params_dir) == (0.0764912345, 0.19532167, 2.5, 20.0)
        physics_app.dtheta_input.setText("")
        physics_app.save_gcms_pressed()                  # refused: GCMS.txt unchanged
        assert iio.read_gcms_file(physics_app.params_dir) == (0.0764912345, 0.19532167, 2.5, 20.0)
    finally:
        physics_app.GCMS_input.setText(str(_G))
        physics_app.dtheta_input.setText("0")
        physics_app.source_diameter_input.setText(str(iio.DEFAULT_SOURCE_DIAMETER))


@pytest.mark.gui
def test_reset_in_cms_restores_the_cms_defaults(physics_app, monkeypatch):
    """Reset instrumental function to default values, in CMS: G 0.09, Δθ 0.1,
    d 5 and D 20, written to GCMS.txt and set in the window; the SMS files
    are left alone."""
    shown = []
    monkeypatch.setattr(iio.QMessageBox, "information", lambda *a, **k: shown.append(a[-1]))
    insexp = os.path.join(physics_app.params_dir, "INSexp.txt")
    with open(insexp, "rb") as f:
        sms_before = f.read()
    physics_app.GCMS_input.setText("0.2")
    physics_app.dtheta_input.setText("0.3")
    physics_app.source_diameter_input.setText("1")
    physics_app.MS_fit.setChecked(True)
    try:
        assert iio.reset_instrumental_defaults(physics_app)
        assert iio.read_gcms_file(physics_app.params_dir) == (0.09, 0.1, 5.0, 20.0)
        assert (iio.get_internal_gcms(physics_app), iio.get_internal_dtheta(physics_app),
                iio.get_cms_diameters(physics_app)) == (0.09, 0.1, (5.0, 20.0))
        assert physics_app.GCMS == 0.09 and shown
        with open(insexp, "rb") as f:
            assert f.read() == sms_before
    finally:
        physics_app.SMS_fit.setChecked(True)
        physics_app.GCMS_input.setText(str(_G))
        physics_app.dtheta_input.setText("0")


def _drive_dialog(monkeypatch, accept=True, edits=None):
    """A dialog with its exec replaced: what it shows is recorded, its boxes
    edited in order, then accepted or rejected. Returns (shown, warnings)."""
    from PySide6.QtWidgets import QDialog, QLabel, QLineEdit
    import syncmoss.supp_menu as supp_menu
    shown = {}

    def fake_exec(dialog):
        editors = dialog.findChildren(QLineEdit)
        shown['values'] = [e.text() for e in editors]
        shown['text'] = " ".join(label.text() for label in dialog.findChildren(QLabel))
        for i, text in (edits or {}).items():
            editors[i].setText(text)
        return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    warned = []
    monkeypatch.setattr(supp_menu.QMessageBox, "warning", lambda *a, **k: warned.append(a[-1]))
    return shown, warned


@pytest.mark.gui
def test_r_over_r_is_the_supp_setting_of_two_diameters(physics_app, monkeypatch):
    """Supp -> Set CMS source and aperture diameters d and D: prefilled with
    the 5 and 20 mm defaults, it says what the aperture is, that only the ratio
    matters and that it matters little; d/D is the model's ratio, saved with
    the defaults (G and Δθ as saved)."""
    from syncmoss.supp_menu import open_cms_diameters_dialog
    assert "Set CMS source and aperture diameters" in [a.text() for a in physics_app.supp_menu.actions()]
    assert iio.get_internal_r_ratio(physics_app) == 0.25
    physics_app.GCMS_input.setText("0.2")                   # a field not saved: the file keeps its G
    shown, warned = _drive_dialog(monkeypatch, edits={0: "1", 1: "0.1"})   # a 1 mm source, a 100 um sample
    try:
        assert open_cms_diameters_dialog(physics_app) and not warned
        assert shown['values'] == ["5.0", "20.0"]
        assert "usually the sample" in shown['text']
        assert "Only their ratio d/D matters" in shown['text'] and "small effect" in shown['text']
        assert iio.get_cms_diameters(physics_app) == (1.0, 0.1) and iio.get_internal_r_ratio(physics_app) == 10.0
        assert iio.read_gcms_file(physics_app.params_dir) == (_G, 0.0, 1.0, 0.1)
        physics_app.source_diameter_input.setText("0")      # a point source
        assert iio.get_internal_r_ratio(physics_app) == 0.0
        _drive_dialog(monkeypatch, accept=False, edits={0: "3"})
        assert not open_cms_diameters_dialog(physics_app)   # cancelled: nothing set
        assert iio.get_cms_diameters(physics_app) == (0.0, 0.1)
    finally:
        physics_app.GCMS_input.setText(str(_G))
        physics_app.source_diameter_input.setText(str(iio.DEFAULT_SOURCE_DIAMETER))
        physics_app.aperture_diameter_input.setText(str(iio.DEFAULT_APERTURE_DIAMETER))


@pytest.mark.gui
@pytest.mark.parametrize("index, text", [(0, "-1"), (1, "0"), (1, "x")])
def test_the_diameters_dialog_refuses_what_it_cannot_use(physics_app, monkeypatch, index, text):
    from syncmoss.supp_menu import open_cms_diameters_dialog
    _, warned = _drive_dialog(monkeypatch, edits={index: text})
    assert not open_cms_diameters_dialog(physics_app) and warned
    assert iio.get_cms_diameters(physics_app) == (5.0, 20.0)


class _Signal:
    def connect(self, *args):
        pass


def _record_searches(monkeypatch):
    """InstrumentalThread replaced: each search started is recorded as
    (ref, mode, dtheta_search) and nothing runs."""
    import syncmoss.syncmoss_main as sm
    started = []

    class _Thread:
        def __init__(self, app, ref, mode, pool, request, dtheta_search=None):
            started.append((ref, mode, dtheta_search))
            self.finished, self.error = _Signal(), _Signal()

        def start(self):
            pass

    monkeypatch.setattr(sm, 'InstrumentalThread', _Thread)
    return started


@pytest.mark.gui
def test_cms_has_refine_only(physics_app, monkeypatch):
    """In CMS mode Refine runs the CMS search -- G fitted, Δθ held (ref 1) --
    whatever SMS approximation is selected, the single line too; Find starts
    nothing and says to use Refine. SMS Find stays Find."""
    started = _record_searches(monkeypatch)
    physics_app.path_list = [physics_app.calibration_path]
    physics_app.MS_fit.setChecked(True)
    try:
        for method in ('gauss', 'theory'):
            physics_app.instrumental_method = method
            for mode in (0, 1, 2):
                physics_app.inprogress = False
                physics_app.instrumental_pressed(physics_app.instrumental_ref(False), mode)
                assert not physics_app.inprogress
                assert "In CMS use Refine" in physics_app.log.toPlainText()
            physics_app.inprogress = False
            physics_app.instrumental_pressed(physics_app.instrumental_ref(True), 0)
        assert started == [(1, 0, None)] * 2
        physics_app.inprogress = False
        physics_app.SMS_fit.setChecked(True)
        physics_app.instrumental_method = 'gauss'
        physics_app.instrumental_pressed(physics_app.instrumental_ref(False), 2)
        assert started[-1] == (0, 2, None)               # SMS Find stays Find
    finally:
        physics_app.inprogress = False
        physics_app.instrumental_method = 'gauss'
        physics_app.SMS_fit.setChecked(True)


@pytest.mark.gui
def test_dtheta_entries(physics_app, monkeypatch):
    """Refine Δθ: pure α-Fe asks for the source's numbers, then fits Δθ; Refine
    Δθ: model takes G from its field; in SMS neither applies. A cancelled
    dialog starts nothing. There is no Find Δθ: nothing to find afresh."""
    import syncmoss.syncmoss_main as sm
    started = _record_searches(monkeypatch)
    answers = []
    monkeypatch.setattr(sm, 'open_dtheta_alpha_fe_dialog', lambda app: answers.pop(0))
    texts = [a.text() for a in physics_app.instrumental_menu.actions()]
    assert {"Refine Δθ: pure α-Fe", "Refine Δθ: model"} <= set(texts)
    assert not [t for t in texts if t.startswith("Find Δθ")]
    physics_app.path_list = [physics_app.calibration_path]
    source = {'G': 0.08, 'f_LM': 0.72, 'C_high': 30.0, 'C_static': 5.0}
    try:
        physics_app.SMS_fit.setChecked(True)
        for mode in (2, 1):
            physics_app.dtheta_pressed(mode)
            assert "not applicable for SMS" in physics_app.log.toPlainText()
        physics_app.MS_fit.setChecked(True)
        answers.extend([None, source])
        physics_app.dtheta_pressed(2)                            # cancelled
        assert started == [] and not physics_app.inprogress
        physics_app.dtheta_pressed(2)
        physics_app.inprogress = False
        physics_app.params_table.select_model(1, 'Singlet')
        physics_app.dtheta_pressed(1)
        assert started == [(1, 2, source), (1, 1, {'G': _G})]
    finally:
        physics_app.inprogress = False
        physics_app.SMS_fit.setChecked(True)


@pytest.mark.gui
def test_resolved_cms_carries_the_smearing_and_sms_none(physics_app, tmp_path):
    physics_app.dtheta_input.setText("0.195")
    physics_app.source_diameter_input.setText("2.5")                # behind the 20 mm aperture
    physics_app.MS_fit.setChecked(True)
    internal = iio.resolve_instrumental_for_file(physics_app, None)
    assert internal['dtheta'] == pytest.approx(0.195) and "Δθ = 0.195" in internal['note']
    assert internal['r_ratio'] == 0.125 and "d/D = 0.125" in internal['note']
    assert iio.build_dat_metadata_lines(physics_app) == ([f"#@GCMS {_G} 0.195 2.5 20.0"], 'CMS')
    # a .dat decides for itself: its numbers, or G alone in an old one (no smearing)
    for header, dtheta, r_ratio in (("#@GCMS 0.123 0.21 1.5 5.0", 0.21, 0.3), ("#@GCMS 0.123", 0.0, 0.25)):
        dat = _write_dat(tmp_path, f"cms{dtheta}_{r_ratio}.dat", physics_app.calibration_path, [header])
        from_file = iio.resolve_instrumental_for_file(physics_app, dat)
        assert (from_file['INS'], from_file['dtheta'], from_file['r_ratio']) == (0.123, dtheta, r_ratio)

    physics_app.SMS_fit.setChecked(True)
    assert iio.resolve_instrumental_for_file(physics_app, None)['dtheta'] == 0.0
    assert iio.resolve_instrumental_for_file(physics_app, None)['r_ratio'] == 0.0
    sms = _write_dat(tmp_path, "sms.dat", physics_app.calibration_path,
                     ["#@INSexp " + " ".join(str(v) for v in _INS_SMS), f"#@INSint {_MULCO_SMS} {_X0_SMS}"])
    assert iio.resolve_instrumental_for_file(physics_app, sms)['dtheta'] == 0.0

    # no numbers: GCMS.txt's -- the frozen one holds G alone: Δθ 0, the default
    # diameters (d/D 0.25)
    physics_app.dtheta_input.setText("")
    physics_app.source_diameter_input.setText("")
    assert iio.get_internal_dtheta(physics_app) == 0.0 and iio.get_internal_r_ratio(physics_app) == 0.25
    physics_app.MS_fit.setChecked(True)
    assert "Δθ" not in iio.resolve_instrumental_for_file(physics_app, None)['note']
    physics_app.dtheta_input.setText("0")
    physics_app.source_diameter_input.setText(str(iio.DEFAULT_SOURCE_DIAMETER))
    physics_app.SMS_fit.setChecked(True)


# --- the CMS searches: G from a single line, Δθ from pure α-Fe --------------------

_SOURCE = {'G': 0.0765, 'f_LM': 0.7, 'C_high': 300.0, 'C_static': 100.0}


def _free(p, fix):
    return sorted(set(range(len(p))) - {int(i) for i in fix})


def test_cms_single_line_reference():
    """Refine: single line in CMS -- a Singlet at the natural width, no
    Gaussian, its T and shift free from 5 and -0.1; Ns and Nnr free as for
    alpha-Fe; no Be window."""
    B = np.full(64, 1000.0)
    model, p, bounds, fix, extra = iio.build_reference_model(None, 0, B, CMS_ch=1)
    assert model == ['Singlet'] and list(p[NB:]) == [5, -0.1, NAT_WIDTH, 0]
    assert _free(p, fix) == [0, 4, NB, NB + 1]
    assert (p[0], p[4]) == pytest.approx((600.0, 400.0))
    assert bounds[0][0] == 0 and bounds[0][4] == 0 and bounds[0][NB] > 0


def test_dtheta_alpha_fe_reference_takes_the_non_resonant_baseline_from_the_source():
    """Ns, its curvature c²s, T and A free; Nnr = Ns (1 - f_LM)/f_LM + C_high
    -- the source's recoil and high-energy quanta -- linked to Ns's curvature,
    as the table links them; the static noise added flat, after the model."""
    from syncmoss.constants import ALPHA_FE_FIELD
    f, c_high, c_static = _SOURCE['f_LM'], _SOURCE['C_high'], _SOURCE['C_static']
    model, p, bounds, fix, extra = iio.dtheta_alpha_fe_reference(np.full(64, 1e4), _SOURCE)
    assert model == ['Sextet'] and _free(p, fix) == [0, 2, NB, NB + 8]
    assert (p[NB], p[NB + 3], p[NB + 4], p[NB + 8]) == (10, ALPHA_FE_FIELD, NAT_WIDTH, 0)
    assert extra['offset'] == c_static
    assert p[0] + p[4] + c_static == pytest.approx(1e4)      # the spectrum's baseline to start
    p[0], p[2] = 5000.0, 3.0
    p = iio.apply_links(p, extra['confu'], extra['Expr'], extra['NExpr'])   # as minimi_hi does
    assert p[4] == pytest.approx(5000.0 * (1 - f) / f + c_high, rel=1e-14)
    assert (p[5], p[6], p[7]) == (p[1], p[2], p[3])           # O, c² and lin of Nnr: Ns's
    v = np.linspace(-11.0, 11.0, 23)
    N0 = p[0] * (1 + p[2] / 1e4 * (v - p[1]) ** 2)
    N1 = p[4] * (1 + p[6] / 1e4 * (v - p[5]) ** 2)
    np.testing.assert_allclose(N1 / p[4], N0 / p[0], rtol=1e-14)


@pytest.mark.parametrize("change, words", [
    ({'f_LM': 1.0}, "between 0 and 1"), ({'f_LM': 0.0}, "between 0 and 1"),
    ({'C_static': -1.0}, "negative"), ({'C_high': 9000.0, 'C_static': 1000.0}, "not below")])
def test_dtheta_alpha_fe_reference_refuses_numbers_it_cannot_use(change, words):
    with pytest.raises(iio.SearchRefused, match=words):
        iio.dtheta_alpha_fe_reference(np.full(64, 1e4), dict(_SOURCE, **change))


@pytest.mark.gui
def test_dtheta_dialog_asks_for_the_source(physics_app, monkeypatch):
    """The "Refine Δθ: pure α-Fe" dialog, its boxes G, f_LM, C_high, static
    noise: G prefilled from its field (every digit), f_LM 0.7, C_high and the
    static noise 0; it says what to find first, and how."""
    from syncmoss.supp_menu import open_dtheta_alpha_fe_dialog
    physics_app.GCMS_input.setText("0.0764912345")
    shown, warned = _drive_dialog(monkeypatch)
    assert open_dtheta_alpha_fe_dialog(physics_app) == {'G': 0.0764912345, 'f_LM': 0.7, 'C_high': 0.0,
                                                       'C_static': 0.0}
    assert shown['values'] == ["0.0764912345", "0.7", "0", "0"] and not warned
    for words in ("single-line", "100 µm Cu", "Static noise", "flat"):
        assert words in shown['text']
    _drive_dialog(monkeypatch, edits={1: "0.75", 2: "120.5", 3: "8"})
    assert open_dtheta_alpha_fe_dialog(physics_app) == {'G': 0.0764912345, 'f_LM': 0.75, 'C_high': 120.5,
                                                       'C_static': 8.0}
    _drive_dialog(monkeypatch, accept=False)
    assert open_dtheta_alpha_fe_dialog(physics_app) is None


@pytest.mark.gui
@pytest.mark.parametrize("index, text", [(0, "-0.01"), (1, "1"), (1, "0"), (2, "-5"), (3, "x")])
def test_dtheta_dialog_refuses_a_number_it_cannot_use(physics_app, monkeypatch, index, text):
    from syncmoss.supp_menu import open_dtheta_alpha_fe_dialog
    _, warned = _drive_dialog(monkeypatch, edits={index: text})
    assert open_dtheta_alpha_fe_dialog(physics_app) is None
    assert warned, "the user should have been told why"


def _synthetic_cms(tmp_path, model, p, v, G, dtheta, r_ratio, JN, offset=0.0):
    """A CMS spectrum of *model* at *p* with the source G, Δθ, r/R (plus a flat
    *offset*), written as a plain two-column .dat (no #@GCMS: the fields decide)."""
    Norm = iio.compute_norm(_SerialPool(), JN, dict(x0=0.0, MulCo=MulCoCMS, INS=G, Met=1))
    counts = np.asarray(m5.TI(v, p, model, JN, _SerialPool(), 0.0, MulCoCMS, G, [0], [0], Met=1, Norm=Norm,
                              dtheta=dtheta, v_center=iio.drive_rest_velocity(v), r_ratio=r_ratio), dtype=float)
    path = os.path.join(str(tmp_path), "synthetic.dat")
    np.savetxt(path, np.column_stack([v, counts + offset]))
    return path


def _cms_search_app(app, path, G, dtheta, r_ratio):
    """*app* in CMS with G and Δθ in their fields and r/R as a source behind a
    10 mm aperture."""
    app.MS_fit.setChecked(True)
    app.jn0_input.setText("32")                        # the spectrum's: the final pass is exact
    app.GCMS_input.setText(repr(G))
    app.dtheta_input.setText(repr(dtheta))
    app.source_diameter_input.setText(repr(10.0 * r_ratio))
    app.aperture_diameter_input.setText("10.0")
    app.path_list = [path]
    assert app.initialize_parameters()
    return app


@pytest.mark.gui
def test_refine_single_line_in_cms_finds_g(physics_app, tmp_path):
    """A single line made with G 0.09 (Δθ 0.1 held, a point source): found
    from G 0.12, and G and Δθ written as the defaults, with the diameters."""
    p = np.zeros(NB)
    p[0], p[4] = 6000.0, 4000.0
    p = np.concatenate([p, [3.0, -0.12, NAT_WIDTH, 0.0]])
    v = np.linspace(-4.0, 4.4, 128)
    path = _synthetic_cms(tmp_path, ['Singlet'], p, v, 0.09, 0.1, 0.0, 32)
    app = _cms_search_app(physics_app, path, 0.12, 0.1, 0.0)
    try:
        result = iio.instrumental(app, 1, 0, pool=_SerialPool())
        assert result['fitted'] == 'G' and result['dtheta'] == 0.1
        assert result['G'] == pytest.approx(0.09, rel=1e-3)          # the minimiser's stop, not the model
        assert result['p'][NB:NB + 2] == pytest.approx([3.0, -0.12], rel=1e-3)
        assert iio.read_gcms_file(app.params_dir) == (result['G'], 0.1, 0.0, 10.0)
    finally:
        app.SMS_fit.setChecked(True)


def _fix_box(pt, row, col):
    return pt.row_widgets[row].layout().itemAt(col + 1).widget().layout().itemAt(0).layout().itemAt(1).widget()


@pytest.mark.gui
def test_cms_searches_with_the_model_over_nbaseline_spectra(physics_app, tmp_path):
    """Refine Δθ: model, then Refine: model (G), with Nbaseline rows: two
    spectra, each its own velocity range -- its own drive-at-rest velocity --
    and baseline, ONE source: Δθ and then G found, every spectrum smeared from
    its own centre."""
    from syncmoss import model_io
    from syncmoss.constants import ALPHA_FE_FIELD
    app = physics_app
    axes = [np.linspace(-11.0, 11.4, 64), np.linspace(-9.0, 8.0, 64)]   # at rest at +0.2, -0.5
    paths = [os.path.join(str(tmp_path), f"cms_{k}.dat") for k in (1, 2)]
    for path, v in zip(paths, axes):
        np.savetxt(path, np.column_stack([v, np.full_like(v, 1e4)]))
    app.process_path.setPlainText(repr(paths))
    app.path_list = paths
    pt = app.params_table
    for row in (1, 3):
        pt.select_model(row, 'Sextet')
    pt.select_model(2, 'Nbaseline')
    for row, ns, thickness in ((0, '10000', None), (2, '8000', None), (1, None, '8'), (3, None, '5')):
        model_io._set_row_param(pt.row_widgets[row], 0, ns or thickness, '0', '', 'False')
        if thickness:
            model_io._set_row_param(pt.row_widgets[row], 3, str(ALPHA_FE_FIELD), '', '', 'True')
        for col in range(1, NB if ns else 14):
            _fix_box(pt, row, col).setChecked(True)          # free: Ns or T only
    G, dtheta, r_ratio = 0.0765, 0.2, 0.25
    _cms_search_app(app, paths[0], G, 0.0, r_ratio)
    app.path_list = paths
    model, p, con1, con2, con3, Distri, Cor, Expr, NExpr, _, Recon, _ = model_io.read_model(app)
    p = iio.apply_links(p, iio._links_of(con1, con2, con3), Expr, NExpr)
    v = np.concatenate(axes)
    Norm = iio.compute_norm(_SerialPool(), app.JN0, dict(x0=0.0, MulCo=MulCoCMS, INS=G, Met=1))
    counts = np.asarray(m5.TI(v, p, model, app.JN0, _SerialPool(), 0.0, MulCoCMS, G, Distri, Cor, Met=1,
                              Norm=Norm, Recon=Recon, lengths=[64, 64], dtheta=dtheta,
                              v_center=[iio.drive_rest_velocity(a) for a in axes], r_ratio=r_ratio), dtype=float)
    for path, a, c in zip(paths, axes, np.split(counts, 2)):
        np.savetxt(path, np.column_stack([a, c]))
    try:
        request = app._instrumental_search_request(1)
        assert request['kind'] == 'nbaseline'
        result = iio.instrumental(app, 1, 1, pool=_SerialPool(), request=request, dtheta_search={'G': G})
        assert result['model'] == ['Sextet', 'Nbaseline', 'Sextet'] and result['lengths'] == [64, 64]
        assert result['fitted'] == 'dtheta' and result['dtheta'] == pytest.approx(dtheta, rel=1e-3)
        app.dtheta_input.setText(repr(dtheta))
        app.GCMS_input.setText("0.1")
        result = iio.instrumental(app, 1, 1, pool=_SerialPool(), request=app._instrumental_search_request(1))
        assert result['fitted'] == 'G' and result['G'] == pytest.approx(G, rel=1e-3)
    finally:
        app.SMS_fit.setChecked(True)


@pytest.mark.gui
def test_refine_dtheta_without_smearing(physics_app, tmp_path):
    """Pure alpha-Fe with no smearing (Δθ 0): the search cannot start at 0 --
    the spectrum changes as Δθ^4 there -- so it starts at 0.15 and goes down,
    here to the bound 0 itself, fitting the spectrum as exactly as Δθ 0 does."""
    model, p, _, _, extra = iio.dtheta_alpha_fe_reference(np.full(8, 1e4), _SOURCE)
    p[0], p[2], p[NB], p[NB + 8] = 7000.0, 2.0, 8.0, 0.3
    p = iio.apply_links(p, extra['confu'], extra['Expr'], extra['NExpr'])
    v = np.linspace(-11.0, 11.4, 128)
    path = _synthetic_cms(tmp_path, model, p, v, _SOURCE['G'], 0.0, 0.5, 32, offset=_SOURCE['C_static'])
    app = _cms_search_app(physics_app, path, 0.1, 0.0, 0.5)
    try:
        result = iio.instrumental(app, 1, 2, pool=_SerialPool(), dtheta_search=_SOURCE)
        dip = result['B'].max() - result['B'].min()
        assert result['dtheta'] < 0.05                     # below that the spectrum cannot tell
        assert np.abs(result['B'] - result['F']).max() < 1e-4 * dip
        assert result['p'][[0, 2, NB, NB + 8]] == pytest.approx([7000.0, 2.0, 8.0, 0.3], rel=1e-3)
    finally:
        app.SMS_fit.setChecked(True)


@pytest.mark.gui
def test_refine_dtheta_from_pure_alpha_fe_finds_it(physics_app, tmp_path):
    """Pure alpha-Fe made with Δθ 0.2 from a disc source (d/D 0.5) and the
    dialog's baseline, its static noise flat on top: found from 0.15 (the field
    at 0) with G held at the dialog's, and G and Δθ written as the defaults."""
    model, p, _, _, extra = iio.dtheta_alpha_fe_reference(np.full(8, 1e4), _SOURCE)
    p[0], p[2], p[NB], p[NB + 8] = 7000.0, 2.0, 8.0, 0.3
    p = iio.apply_links(p, extra['confu'], extra['Expr'], extra['NExpr'])
    v = np.linspace(-11.0, 11.4, 128)                  # the drive at rest at +0.2 mm/s
    path = _synthetic_cms(tmp_path, model, p, v, _SOURCE['G'], 0.2, 0.5, 32, offset=_SOURCE['C_static'])
    app = _cms_search_app(physics_app, path, 0.1, 0.0, 0.5)
    try:
        result = iio.instrumental(app, 1, 2, pool=_SerialPool(), dtheta_search=_SOURCE)
        assert result['fitted'] == 'dtheta' and result['G'] == _SOURCE['G']
        assert result['dtheta'] == pytest.approx(0.2, rel=1e-3)
        assert result['p'][[0, 2, NB, NB + 8]] == pytest.approx([7000.0, 2.0, 8.0, 0.3], rel=1e-3)
        assert result['baseline_offset'] == _SOURCE['C_static']
        assert iio.read_gcms_file(app.params_dir) == (_SOURCE['G'], result['dtheta'], 5.0, 10.0)
        # the window takes both into the fields, even after a switch to SMS meanwhile
        app.SMS_fit.setChecked(True)
        app.on_instrumental_finished(result)
        assert (iio.get_internal_gcms(app), iio.get_internal_dtheta(app)) == (_SOURCE['G'], result['dtheta'])
        assert "Δθ found" in app.log.toPlainText()
    finally:
        app.SMS_fit.setChecked(True)
