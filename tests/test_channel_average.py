"""The channel average (models, "Channel average"; Supp -> "Average model over
each channel", ON for the user, OFF for the rest of the suite: conftest):
every point of a spectrum is the model averaged over the velocities the drive
runs through while the point is counted. Averaging the model over a box is
averaging the SOURCE over it, so TI's callers prepare it with the source
(instrumental_io.source_for: models.source_smearing, the source line at every
node as each point sees it, and its Norm, one per point), as for the cosine
smearing -- and with it, inside the average over the rays. The widths come from
the drive, fitted once to a spectrum's whole axis (models.fit_drive), as a
function of the velocity alone.

Pinned: the drive of folded sine and triangular axes (every fold phase, any
order of the points), of raw calibration sweeps, and an axis with points
missing refused; the box averages of the CMS pseudo-Voigt, the SMS sum of
Gaussians and the theoretical shape against fine sums; the BRUTE FORCE -- TI
with the average against the exact transmission integral averaged over each
channel in time (and over the rays), triangular and sine drives, CMS without
and with the cosine smearing, SMS with the Gaussians and with the theoretical
shape: the average adds no error to TI's own, and what it changes is far
larger; off is the former model, bit for bit; a part of the points; Nbaseline
sections; the callers that pass it on; the Supp switch and its warning.
Serial pool.
"""
import os

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import syncmoss.models as m5
import syncmoss.sms_theory as smst
from syncmoss import instrumental_io as iio
from syncmoss.Calibration import MulCoCMS, _fold_sinusoidal, _fold_triangular, _raw_drive
from syncmoss.constants import NAT_WIDTH, number_of_baseline_parameters as NB

from conftest import FROZEN_PARAMETERS, SerialPool


def _frozen(name, delimiter=' '):
    return np.genfromtxt(os.path.join(FROZEN_PARAMETERS, name), delimiter=delimiter)


_G = float(_frozen("GCMS.txt", delimiter='\t'))
_MULCO, _X0 = (float(v) for v in _frozen("INSint.txt"))
_METHODS = {
    "CMS": dict(method='CMS', Met=1, INS=_G, x0=0.0, MulCo=MulCoCMS, dtheta=0.0,
                r_ratio=iio.DEFAULT_SOURCE_DIAMETER / iio.DEFAULT_APERTURE_DIAMETER),
    "SMS Gaussians": dict(method='SMS', Met=0, INS=np.atleast_1d(_frozen("INSexp.txt")), x0=_X0, MulCo=_MULCO,
                          dtheta=0.0, r_ratio=0.0),
    "SMS theory": dict(method='SMS', Met=0, INS=np.atleast_1d(_frozen("INSth.txt")), x0=_X0, MulCo=_MULCO,
                       dtheta=0.0, r_ratio=0.0),
}
assert smst.ins_kind(_METHODS["SMS theory"]["INS"]) != smst.KIND_GAUSS
_DTHETA = 0.25                                    # rad, with the default d/D


@pytest.fixture(autouse=True)
def _empty_store():
    m5._NODE_SUMS.clear()
    yield
    m5._NODE_SUMS.clear()


def _singlets(lines, N0=1.0):
    """Baseline (N0 only) + natural-width singlets [(position, thickness), ...]."""
    p = np.zeros(NB)
    p[0] = N0
    return np.concatenate([p] + [[t, x, NAT_WIDTH, 0.0] for x, t in lines])


def _method(name, x, dtheta=0.0, averaged=True, **changes):
    """The resolved method *name* with the drive values of the whole axis x."""
    mp = dict(_METHODS[name], v_center=iio.drive_rest_velocity(x), drive=m5.fit_drive(x) if averaged else None,
              **changes)
    if dtheta:
        mp['dtheta'] = dtheta
    return mp


def _ti(x, p, model, mp, JN, lengths=None):
    """TI with the source source_for prepares from *mp*, as every caller does."""
    norm, smearing = iio.source_for(SerialPool(), x, JN, mp)
    return np.asarray(m5.TI(np.asarray(x, dtype=float), p, list(model), JN, SerialPool(), mp['x0'], mp['MulCo'],
                            mp['INS'], [0], [0], Met=mp['Met'], Norm=norm, lengths=lengths, smearing=smearing),
                      dtype=float)


# --- the drives: a folded half period of 2*256 channels, at rest at +0.15 ------------

_V_C = 0.15
_POINTS = 256
_TIMES = 48                                        # moments sampled in a channel


def _drive(kind, V0):
    """The velocities written for a folded half period (each its channel's time
    average) and the drive's velocity at _TIMES moments of each channel."""
    t = np.arange(_POINTS)[:, None] + (np.arange(_TIMES) + 0.5) / _TIMES
    if kind == "triangular":
        moments = _V_C + V0 - 2 * V0 / _POINTS * t
    else:
        moments = _V_C + V0 * np.cos(np.pi * t / _POINTS)
    return moments.mean(axis=1), moments


# --- the drive and its channel widths -------------------------------------------------

_N_CH = 1024                                       # raw channels, one period
_CHANNELS = np.arange(_N_CH, dtype=float)


def _sin_cal(p):
    """Calibration's sinusoidal drive: every channel's time-averaged velocity."""
    return p[0] / np.pi * _N_CH * np.sin(np.pi / _N_CH) * np.cos(p[1] + np.pi / _N_CH * (2 * _CHANNELS + 1)) + p[2]


def _lin_cal(p):
    """Calibration's triangular drive: two ramps of one step."""
    return np.where(_CHANNELS <= int(_N_CH / 2), p[0] - p[2] * _CHANNELS, p[1] - p[2] * (_N_CH - 1 - _CHANNELS))


def _sine_widths(p, v):
    """The range of velocities of the sine drive *p* in the channel whose time
    average is v: 2 V0 sin(pi/n) |sin(phase)|, the phase from v."""
    mean_amplitude = p[0] / np.pi * _N_CH * np.sin(np.pi / _N_CH)
    return 2 * p[0] * np.sin(np.pi / _N_CH) * np.sqrt(np.clip(1 - ((v - p[2]) / mean_amplitude) ** 2, 0, None))


@pytest.mark.parametrize("phase", [0.0, 0.3, 37.5, 200.25, 511.0])
def test_a_folded_sine_axis_gives_its_drive_whatever_the_fold(phase):
    """The axis the calibration writes -- the raw sweep folded, in up to three
    pieces, with 6 decimals -- in its own order, reversed and shuffled: a sine
    of the drive's centre and phase step, the widest channel as wide as the
    drive's, and every channel twice as narrow as the widest or wider within
    1e-4 of the drive's width at its velocity (a folded point joins two
    channels a fraction of one apart: the turning points tell them apart)."""
    p = [13.48, np.pi / _N_CH * phase, 0.69]
    folded = np.round(_fold_sinusoidal(_sin_cal(p), np.ones(_N_CH), np.ones(_N_CH), p[2])[0], 6)
    widest = 2 * p[0] * np.sin(np.pi / _N_CH)
    for axis in (folded, folded[::-1], np.random.default_rng(1).permutation(folded)):
        drive = m5.fit_drive(axis)
        assert drive[0] == 'sine' and drive[3] == pytest.approx(2 * np.pi / _N_CH, rel=1e-6)
        assert drive[1] == pytest.approx(p[2], abs=1e-5)
        widths = m5.channel_widths(axis, drive)
        assert widths.max() == pytest.approx(widest, rel=1e-5)
        broad = widths > 0.5 * widest
        assert broad.sum() > 0.6 * len(axis)
        assert np.abs(widths[broad] / _sine_widths(p, axis[broad]) - 1).max() < 1e-4


@pytest.mark.parametrize("phase", [0.0, 1.0, 3.0])
def test_a_folded_triangular_axis_gives_its_step(phase):
    """Every channel the step wide -- also at phase 0, where the fold leaves one
    end point half a step off."""
    p = [10.0, 10.0 - 20.0 / 512 * phase, 20.0 / 512]
    folded = np.round(_fold_triangular(_lin_cal(p), np.ones(_N_CH), np.ones(_N_CH), p)[0], 6)
    drive = m5.fit_drive(folded)
    assert drive[0] == 'triangular' and drive[1] == pytest.approx(p[2], rel=1e-4)
    assert np.array_equal(m5.channel_widths(folded[:7], drive), np.full(7, drive[1]))


def test_the_tests_calibration_axis_is_a_sine_of_1024_channels():
    v = np.loadtxt(os.path.join(FROZEN_PARAMETERS, "Calibration.dat"), comments='#', usecols=0)
    drive = m5.fit_drive(v)
    assert drive[0] == 'sine' and drive[3] == pytest.approx(2 * np.pi / 1024, rel=1e-6)
    assert drive[1] == pytest.approx(iio.drive_rest_velocity(v), abs=1e-3)


def test_an_axis_with_points_missing_is_refused():
    """Its points are no longer one channel apart: no drive, and the window
    names the spectrum. Without the average nothing is asked of the axis."""
    v = np.loadtxt(os.path.join(FROZEN_PARAMETERS, "Calibration.dat"), comments='#', usecols=0)
    holed = np.delete(v, np.arange(200, 210))
    for axis in (holed, np.delete(_drive("triangular", 12.0)[0], [100, 101])):
        with pytest.raises(ValueError, match="not those of a triangular or sine drive"):
            m5.fit_drive(axis)

    class App:
        channel_average = True
    with pytest.raises(ValueError, match="cut.dat: the velocities are not"):
        iio.drive_values(App(), holed, "cut.dat")
    App.channel_average = False
    assert iio.drive_values(App(), holed)['drive'] is None


@pytest.mark.parametrize("kind", ["sine", "triangular"])
def test_a_raw_sweep_gives_the_calibration_its_drive(kind):
    """The calibration's raw channels (both visits, in recorded order): the
    sine's every channel width exact; the triangle's step."""
    if kind == "sine":
        p = [9.7, 0.37, 0.21]
        v = _sin_cal(p)
        widths = m5.channel_widths(v, _raw_drive(v))
        assert np.abs(widths - _sine_widths(p, v)).max() < 1e-9 * widths.max()
    else:
        p = [10.0, 10.0 - 20.0 / 512 * 2.5, 20.0 / 512]
        assert _raw_drive(_lin_cal(p)) == ('triangular', pytest.approx(p[2], rel=1e-12))


# --- the boxes ----------------------------------------------------------------------

def _fine_box(f, centre, width, n=4001):
    """f averaged over a box of *width* about *centre*: a midpoint sum."""
    t = (np.arange(n) + 0.5) / n - 0.5
    return np.mean([f(centre + width * tk) for tk in t], axis=0)


def test_voigt_box_is_voigt_averaged_over_the_box():
    """Its cumulative's difference (A&S erf in the Gaussian term: within 1e-5 of
    the peak), for boxes from a hundredth of the line to three widths; below
    _BOX_POINT_LIMIT of the line, the point value itself."""
    gL, gG = NAT_WIDTH * MulCoCMS, _G * MulCoCMS
    S = np.linspace(-0.3, 0.3, 601)
    peak = m5.Voigt(gL, gG, np.array([0.0]))[0]
    for width in (0.01, 0.3, 1.0, 3.0):
        W = width * (gL + gG)
        fine = _fine_box(lambda s: m5.Voigt(gL, gG, s), S, W)
        assert np.abs(m5.Voigt_box(gL, gG, S, W) - fine).max() < 1e-5 * peak, width
    tiny = 0.5 * m5._BOX_POINT_LIMIT * (gL + gG)
    assert np.array_equal(m5.Voigt_box(gL, gG, S, tiny), m5.Voigt(gL, gG, S))


@pytest.mark.parametrize("name", ["SMS Gaussians", "SMS theory", "CMS"])
def test_the_source_lines_are_the_source_averaged_over_each_channel(name):
    """At every integration node, every point's line is the plain source line
    averaged over its channel (a fine sum over the box, MulCo*W wide in the
    source variable): the Gaussians' erf box and the CMS cumulative within 1e-5
    of the peak; the theoretical shape's channel nodes, on its table (linear
    between the tabulated points), within 1.5e-4 -- below that table's own
    error, 5e-4 of the peak between its points."""
    JN = 32
    x, _ = _drive("sine", 12.0)
    mp = _method(name, x)
    W = m5.channel_widths(x, mp['drive'])
    lines, shift = m5.source_smearing(x, JN, mp['MulCo'], mp['INS'], mp['Met'], 0.0, mp['v_center'],
                                      mp['r_ratio'], channel_widths=W, x0=mp['x0'])
    assert shift is None and lines.shape == (JN, len(x))
    EE = m5._node_grid(mp['Met'], JN)
    if name == "CMS":
        cof = m5._cms_map_coefficients(JN)
        s = cof[0] * np.log((1 + EE) / (1 - EE)) + cof[1] * np.log((2 + EE) / (2 - EE)) \
            + cof[2] * np.log((3 + EE) / (3 - EE))

        def plain(t):
            return m5.Voigt(NAT_WIDTH * MulCoCMS, _G * MulCoCMS, t.ravel()).reshape(t.shape)
        tolerance = 1e-5
    else:
        s = mp['MulCo'] * mp['x0'] + np.log((1 + EE) / (1 - EE))

        def plain(t):
            return smst.ins_shape(mp['INS'], t / mp['MulCo']) / mp['MulCo']
        tolerance = 1e-5 if name == "SMS Gaussians" else 1.5e-4
    fine = _fine_box(plain, s[:, None] + 0 * W[None, :], mp['MulCo'] * W[None, :], 801)
    peak = plain(s).max()
    assert np.abs(lines - fine).max() < tolerance * peak
    # and it is not the point source: the widest channels change it by far more
    assert np.abs(lines - plain(s)[:, None]).max() > 20 * tolerance * peak


def test_a_triangular_drive_s_line_is_one_for_every_point():
    """Every channel as wide: a node's line is computed once and is the same at
    every point -- the values those of the line of each point."""
    x, _ = _drive("triangular", 6.0)
    for name in ("CMS", "SMS Gaussians", "SMS theory"):
        mp = _method(name, x)
        W = m5.channel_widths(x, mp['drive'])
        lines, _ = m5.source_smearing(x, 16, mp['MulCo'], mp['INS'], mp['Met'], 0.0, mp['v_center'],
                                      mp['r_ratio'], channel_widths=W, x0=mp['x0'])
        one_by_one, _ = m5.source_smearing(x[:1], 16, mp['MulCo'], mp['INS'], mp['Met'], 0.0, mp['v_center'],
                                           mp['r_ratio'], channel_widths=W[:1], x0=mp['x0'])
        assert np.array_equal(lines, np.repeat(one_by_one, len(x), axis=1)), name


# --- the brute force ----------------------------------------------------------------
# Absorber: natural-width lines of several thicknesses, one of them saturated.

_LINES = [(-3.1, 1.0), (-0.4, 4.0), (0.35, 0.3), (2.6, 8.0)]
_H = 0.002                                         # the exact integral's grid, mm/s
_SPAN = 40.0                                       # how far the source is followed, mm/s
_JN = 64


def _exact(name):
    """The exact N0 = 1 count rate of _LINES when every photon leaves with the
    velocity e: 1 - int S(u) (1 - T(e + u)) du over a fine grid -- no window, no
    Norm; the SMS sources unit area -- as a cubic spline in e."""
    E = np.arange(-12.5 - _SPAN, 12.5 + _SPAN + _H / 2, _H)
    absorbed = 1 - np.exp(-sum(np.pi * NAT_WIDTH / 2 * t * m5.Voigt(NAT_WIDTH, 0.0, E - x) for x, t in _LINES))
    u = np.arange(-_SPAN, _SPAN + _H / 2, _H)
    mp = _METHODS[name]
    if name == "CMS":
        S = m5.Voigt(NAT_WIDTH, _G, u)
    else:
        S = smst.ins_shape(mp['INS'], u)
        S = S / (S.sum() * _H)
    n = len(E) + len(u) - 1
    correlation = np.fft.irfft(np.fft.rfft(absorbed, n) * np.fft.rfft(S[::-1], n), n)
    e = E[0] - u[-1] + np.arange(n) * _H
    inside = np.abs(e) <= 12.5
    return CubicSpline(e[inside], 1 - correlation[inside] * _H)


def _rays(dtheta):
    """The rays' y = 1 - cos(theta) and weights: none, or the disc's rule at
    many more nodes than the model takes (its accuracy: test_cosine_smearing)."""
    if not dtheta:
        return np.zeros(1), np.ones(1)
    y, w = m5.disc_rule(dtheta, _METHODS["CMS"]["r_ratio"], 96)
    return np.asarray(y), np.asarray(w)


@pytest.mark.parametrize("name, kind, dtheta", [
    ("CMS", "triangular", 0.0), ("CMS", "sine", 0.0), ("CMS", "triangular", _DTHETA), ("CMS", "sine", _DTHETA),
    ("SMS Gaussians", "triangular", 0.0), ("SMS Gaussians", "sine", 0.0),
    ("SMS theory", "triangular", 0.0), ("SMS theory", "sine", 0.0)])
def test_the_average_follows_the_brute_force(name, kind, dtheta):
    """Brute force: the exact spectrum of the photons each ray takes from the
    drive's velocity at every moment of the channel (the transducer's velocity,
    the rays from where it is at rest), averaged over the channel and the rays.
    TI with the channel average -- its widths from the drive fitted to the axis
    -- is as close to it as TI without is to the exact spectrum at the written
    velocities: the average adds no error of its own; and what it changes is
    more than 10 times that (24 to 1300)."""
    C = _exact(name)
    x, moments = _drive(kind, 12.0)                # steps up to 0.15 mm/s: wider than a natural line
    y, w = _rays(dtheta)
    point = C(_V_C + (x[:, None] - _V_C) * (1 - y[None, :])) @ w
    brute = (C(_V_C + (moments[:, :, None] - _V_C) * (1 - y[None, None, :])) @ w).mean(axis=1)
    p = _singlets(_LINES)
    model = ['Singlet'] * len(_LINES)
    plain_error = np.abs(_ti(x, p, model, _method(name, x, dtheta, averaged=False), _JN) - point).max()
    error = np.abs(_ti(x, p, model, _method(name, x, dtheta), _JN) - brute).max()
    assert error < 1.25 * plain_error
    assert np.abs(brute - point).max() > 10 * error


# --- off, a part of the points, Nbaseline, the callers --------------------------------

@pytest.mark.parametrize("name, dtheta", [("CMS", 0.0), ("CMS", _DTHETA), ("SMS Gaussians", 0.0),
                                          ("SMS theory", 0.0)])
def test_off_is_the_former_model(name, dtheta):
    """No drive: the source as before -- none prepared without the cosine
    smearing (the plain Norm), the rays alone with it, bit for bit."""
    x, _ = _drive("triangular", 6.0)
    mp = _method(name, x, dtheta, averaged=False)
    norm, smearing = iio.source_for(SerialPool(), x, 32, mp)
    if not dtheta:
        assert smearing is None and norm == iio.compute_norm(SerialPool(), 32, mp)
        return
    EE = m5._node_grid(1, 32)
    cof = m5._cms_map_coefficients(32)
    offsets = cof[0] * np.log((1 + EE) / (1 - EE)) + cof[1] * np.log((2 + EE) / (2 - EE)) \
        + cof[2] * np.log((3 + EE) / (3 - EE))
    U = (x - mp['v_center']) * MulCoCMS
    lines, mean = m5.Voigt_disc(NAT_WIDTH * MulCoCMS, _G * MulCoCMS, offsets[:, None], U[None, :], dtheta,
                                mp['r_ratio'])
    assert np.array_equal(smearing[0], lines) and np.array_equal(smearing[1], U * mean)


@pytest.mark.parametrize("name, dtheta", [("CMS", _DTHETA), ("SMS Gaussians", 0.0), ("SMS theory", 0.0)])
def test_a_part_of_the_points_gives_the_same_values(name, dtheta):
    """The fit on the points exclusion regions leave sees the model of the
    whole spectrum there: the drive is the whole axis's, its widths taken at
    each point's own velocity."""
    x, _ = _drive("sine", 12.0)
    keep = (x < -3.0) | (x > -2.0)
    p = _singlets(_LINES)
    model = ['Singlet'] * len(_LINES)
    mp = _method(name, x, dtheta)
    whole = _ti(x, p, model, mp, 32)
    np.testing.assert_allclose(_ti(x[keep], p, model, mp, 32), whole[keep], rtol=1e-13, atol=0)


def test_each_nbaseline_spectrum_has_its_own_drive():
    """Two spectra of different steps in one Nbaseline model: each averaged as
    it is alone, and not with the other's drive."""
    x1, _ = _drive("triangular", 6.0)
    x2 = np.linspace(-11.0, 11.4, 129)
    p1, p2 = _singlets([(-0.4, 4.0)]), _singlets([(2.6, 2.0)])
    model = ['Singlet', 'Nbaseline', 'Singlet']
    for name in ("CMS", "SMS Gaussians"):
        mps = [_method(name, x1, _DTHETA if name == "CMS" else 0.0), _method(name, x2)]
        pieces = [iio.source_for(SerialPool(), xi, 32, mpi) for xi, mpi in zip((x1, x2), mps)]
        mp = mps[0]
        both = np.asarray(m5.TI(np.concatenate([x1, x2]), np.concatenate([p1, p2]), model, 32, SerialPool(),
                                mp['x0'], mp['MulCo'], mp['INS'], [0], [0], Met=mp['Met'],
                                Norm=[n for n, _ in pieces], lengths=[len(x1), len(x2)],
                                smearing=[s for _, s in pieces]), dtype=float)
        np.testing.assert_allclose(both[:len(x1)], _ti(x1, p1, ['Singlet'], mps[0], 32), rtol=1e-12, atol=0)
        np.testing.assert_allclose(both[len(x1):], _ti(x2, p2, ['Singlet'], mps[1], 32), rtol=1e-12, atol=0)
        # the same, prepared at once for both (source_smearing with lengths)
        widths = np.concatenate([m5.channel_widths(xi, mpi['drive']) for xi, mpi in zip((x1, x2), mps)])
        smearing = m5.source_smearing(np.concatenate([x1, x2]), 32, mp['MulCo'], mp['INS'], mp['Met'],
                                      [mps[0]['dtheta'], 0.0], [mps[0]['v_center'], mps[1]['v_center']],
                                      mp['r_ratio'], [len(x1), len(x2)], widths, mp['x0'])
        for prepared, (_, alone) in zip(smearing, pieces):
            assert np.array_equal(prepared[0], alone[0])
        other = dict(mps[1], drive=mps[0]['drive'])
        assert not np.allclose(_ti(x2, p2, ['Singlet'], other, 32), both[len(x1):], rtol=1e-6, atol=0)


def test_component_curves_hires_check_and_model_spectrum_use_the_average():
    """Each prepares its source from the method's drive (source_for) for its
    own velocities and JN."""
    from syncmoss import fitting_io
    from syncmoss.spectrum_io import compute_model_spectrum
    x, _ = _drive("sine", 6.0)
    p = _singlets([(0.35, 2.0)])
    for name in ("CMS", "SMS Gaussians"):
        mp = _method(name, x, _DTHETA if name == "CMS" else 0.0)
        averaged = _ti(x, p, ['Singlet'], mp, 16)
        assert not np.allclose(averaged, _ti(x, p, ['Singlet'], dict(mp, drive=None), 16), rtol=1e-6)
        FS, _ = fitting_io.compute_component_curves(x, [p], [['Singlet']], [], [], 16, SerialPool(), mp,
                                                    m5.SMS_POL_DEFAULT)
        np.testing.assert_allclose(np.asarray(FS[0], dtype=float), averaged, rtol=1e-12, atol=0)
        hires = np.asarray(iio.hires_model_diff(SerialPool(), 16, x, p, ['Singlet'], mp, averaged), dtype=float)
        hi = _ti(x, p, ['Singlet'], mp, 16 * iio.HIRES_INTEGRATION_FACTOR)
        np.testing.assert_allclose(hires, hi - averaged, rtol=0, atol=1e-12)
        job = dict(A=x, p=p, model=['Singlet'], JN=16, method_params=mp, Distri=[0], Cor=[0], Recon=[0],
                   pol=m5.SMS_POL_DEFAULT)
        np.testing.assert_allclose(compute_model_spectrum(job, SerialPool()), averaged, rtol=1e-12, atol=0)


def test_channel_widths_given_for_other_points_are_refused():
    x, _ = _drive("triangular", 6.0)
    mp = _method("SMS Gaussians", x)
    with pytest.raises(ValueError, match="channel widths given for 256 points, the spectrum has 255"):
        m5.source_smearing(x[1:], 16, mp['MulCo'], mp['INS'], 0, 0.0, 0.0, 0.0,
                           channel_widths=m5.channel_widths(x, mp['drive']), x0=mp['x0'])


# --- the switch --------------------------------------------------------------------

def test_the_shipped_default_is_on(channel_average_off):
    """ON for the user (instrumental_io's and the window's default); the suite
    runs with it OFF (conftest)."""
    assert channel_average_off == (True, True)
    assert iio.DEFAULT_CHANNEL_AVERAGE is False


@pytest.mark.gui
def test_a_spectrum_whose_axis_lacks_points_is_refused_with_a_window(physics_app, tmp_path, monkeypatch):
    """With the average on, Fit and Show model are not started for a spectrum
    whose velocities miss points (saved by another program): a window names it
    and says what to do -- calibrate and save it in SYNCmoss, or turn the
    average off -- and where to report it. With the axis whole, or the average
    off, nothing is asked."""
    import syncmoss.supp_menu as supp_menu
    app = physics_app
    shown = []
    monkeypatch.setattr(supp_menu.QMessageBox, "warning", lambda *args, **kwargs: shown.append(args[-1]))
    A, B = np.loadtxt(os.path.join(FROZEN_PARAMETERS, "Calibration.dat"), comments='#', unpack=True)
    holed, whole = str(tmp_path / "holed.dat"), str(tmp_path / "whole.dat")
    np.savetxt(holed, np.column_stack((np.delete(A, range(200, 210)), np.delete(B, range(200, 210)))), fmt='%.6f')
    np.savetxt(whole, np.column_stack((A, B)), fmt='%.6f')
    app.params_table.select_model(1, 'Sextet')
    app.channel_average = True
    assert app.channel_average_refused([whole, holed], "Fit")
    assert len(shown) == 1 and "holed.dat: the velocities are not those of a triangular or sine drive" in shown[0]
    for words in ("Calibrate and save the spectrum in SYNCmoss", "Supp → Average model over each channel",
                  "Supp → Contact the author"):
        assert words in shown[0]
    assert "Fit was not started — holed.dat" in app.log.toPlainText()
    for press, thread in ((app.fit_pressed, 'fitting_thread'), (app.showM_pressed, 'show_model_thread')):
        app.process_path.setPlainText(repr([holed]))
        press()
        assert getattr(app, thread, None) is None and not app.inprogress
    assert len(shown) == 3
    assert not app.channel_average_refused([whole], "Fit")
    app.channel_average = False
    assert not app.channel_average_refused([holed], "Fit") and len(shown) == 3


@pytest.mark.gui
def test_supp_switches_the_average_and_warns(physics_app, monkeypatch):
    """Supp -> "Average model over each channel" switches it and warns that an
    instrumental function or a calibration found with the other setting does
    not fit it: Keep keeps the switch, Switch back undoes it. The drive values
    every caller adds follow it."""
    import syncmoss.supp_menu as supp_menu
    from syncmoss.supp_menu import channel_average_text
    app = physics_app
    x, _ = _drive("sine", 6.0)
    asked = []
    answers = [True, False, True]
    monkeypatch.setattr(supp_menu, 'keep_channel_average', lambda window, on: asked.append(on) or answers.pop(0))
    assert not iio.channel_average(app)                              # the suite's
    assert iio.drive_values(app, x) == {'v_center': iio.drive_rest_velocity(x), 'drive': None}
    entries = {action.text(): action for action in app.supp_menu.actions()}
    entries[channel_average_text(False)].trigger()                   # ON, kept
    assert iio.channel_average(app) and app.channel_average_action.text() == channel_average_text(True)
    assert iio.drive_values(app, x)['drive'] == m5.fit_drive(x)
    assert "each point averaged over its channel" in app.log.toPlainText()
    app.channel_average_action.trigger()                             # OFF, then switched back: ON
    assert iio.channel_average(app) and app.channel_average_action.text() == channel_average_text(True)
    app.channel_average_action.trigger()                             # OFF, kept
    assert not iio.channel_average(app) and app.channel_average_action.text() == channel_average_text(False)
    assert "each point at its velocity" in app.log.toPlainText()
    assert asked == [True, False, False]
