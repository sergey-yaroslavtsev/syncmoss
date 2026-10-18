"""CMS_source_geometry_calibration: the drive, G, Δθ and f_LM fitted together
on an alpha-Fe and a single-line raw spectrum (docs/cosine_smearing.md, §7).

Fast: the start values, the refusals, the count distortion, and the joint
model against each spectrum computed on its own. Slow (marked): the whole
procedure on synthetic raw spectra of a known drive and source, in a
throw-away parameters folder.
"""
import os

import numpy as np
import pytest

from syncmoss import CMS_source_geometry_calibration as sc
from syncmoss.constants import ALPHA_FE_FIELD, NAT_WIDTH

from conftest import SerialPool


def _write_mca(path, counts):
    with open(path, "w") as f:
        f.write(f"#S 1 test\n#@CHANN {len(counts)} 0 {len(counts) - 1} 1\n@A ")
        f.write(" ".join(str(int(c)) for c in counts) + "\n#\n")
    return str(path)


def _triangle(n, vmax, rest):
    """A triangular drive as the calibration's fixtures have it: from vmax down
    and back, the drive at rest at *rest* on the alpha-Fe axis."""
    ch, half = np.arange(n), n // 2
    step = 2 * vmax / half
    return np.where(ch < half, vmax - step * ch, vmax - step * (n - 1 - ch)) + rest


def test_unusable_fields_start_from_the_usual_source():
    assert sc.start_values(0.0765, 0.2) == (0.0765, 0.2)
    assert sc.start_values(0.0, 0.0) == (sc.START_G, sc.START_DTHETA)
    assert sc.start_values("x", -0.1) == (sc.START_G, sc.START_DTHETA)
    assert sc.start_values(1.5, 2.0) == (sc.START_G, sc.START_DTHETA)


def test_it_needs_two_raw_spectra_of_one_drive(tmp_path):
    a = _write_mca(tmp_path / "a.mca", np.full(64, 100))
    b = _write_mca(tmp_path / "b.mca", np.full(64, 100))
    c = _write_mca(tmp_path / "c.mca", np.full(32, 100))
    dat = tmp_path / "d.dat"
    dat.write_text("0 1\n")
    assert "two raw spectra" in sc.refusal([a])
    assert "not a raw spectrum" in sc.refusal([a, str(dat)])
    assert "does not exist" in sc.refusal([a, str(tmp_path / "none.mca")])
    assert "64 and 32 channels" in sc.refusal([a, c])
    assert sc.refusal([a, b]) is None


def test_the_source_needs_f_lm_and_both_spectra_s_counts():
    good = {'f_LM': 0.7, 'C_high': (100.0, 50.0), 'C_static': (10.0, 5.0)}
    assert sc.source_refusal(good) is None
    assert "between 0 and 1" in sc.source_refusal(dict(good, f_LM=1.0))
    assert "both spectra" in sc.source_refusal(dict(good, C_high=(100.0,)))
    assert "negative" in sc.source_refusal(dict(good, C_static=(10.0, -1.0)))


def test_the_distortion_is_one_parabola_per_half_sweep():
    n = 16
    assert np.all(sc.distortion(n, [0, 0, 0, 0, 0]) == 1)
    w = sc.distortion(n, [1e-3, 3.0, 0.01, 2e-3, 12.0])
    ch = np.arange(n)
    np.testing.assert_allclose(w[:8], 1 + 1e-3 * (ch[:8] - 3) ** 2 + 0.01)
    np.testing.assert_allclose(w[8:], 1 + 2e-3 * (ch[8:] - 12) ** 2)


def test_both_spectra_together_are_each_on_its_own():
    """raw_counts of both spectra = alpha-Fe alone and the single line alone,
    each on the drive's velocities times the distortion plus its own static
    noise, its Nnr = Ns (1 - f_LM)/f_LM + its own C_high: the joint model's
    sections (Nbaseline) hold the right parameters."""
    n, JN, r_ratio = 64, 32, 0.25
    source = {'f_LM': 0.7, 'C_high': (900.0, 400.0), 'C_static': (30.0, 20.0)}
    q = np.zeros(sc.N_FIT)
    q[sc.DRIVE] = [6.2, 6.2, 0.39]                     # Calibration.lin_cal's: v at 0, at the last, step
    q[sc.DISTORTION] = [2e-6, 20.0, 1e-3, 3e-6, 40.0]
    q[sc.G_I], q[sc.U_I], q[sc.F_I] = 0.08, 1 - np.cos(0.2), 0.72
    q[[sc.ALPHA_T, sc.ALPHA_A, sc.ALPHA_NS]] = [6.0, 0.1, 7e4]
    q[[sc.LINE_T, sc.LINE_SHIFT, sc.LINE_NS]] = [12.0, -0.1, 5e4]

    def velocity(ch, d):
        return np.where(ch <= n // 2, d[0] - d[2] * ch, d[1] - d[2] * (n - 1 - ch))
    pool = SerialPool()
    joint = sc.raw_counts(q, velocity, n, source, r_ratio, JN, pool)
    v = velocity(np.arange(n, dtype=float), q[sc.DRIVE])
    w = sc.distortion(n, q[sc.DISTORTION])
    dtheta = np.arccos(1 - q[sc.U_I])
    alpha_p, line_p = sc.alpha_fe_parameters(q, 900.0), sc.single_line_parameters(q, 400.0)
    assert alpha_p[4] == pytest.approx(7e4 * 0.28 / 0.72 + 900.0)
    assert line_p[4] == pytest.approx(5e4 * 0.28 / 0.72 + 400.0)
    alpha = sc.counts(v, alpha_p, ['Sextet'], None, q[sc.G_I], dtheta, r_ratio, JN, pool) * w + 30.0
    line = sc.counts(v, line_p, ['Singlet'], None, q[sc.G_I], dtheta, r_ratio, JN, pool) * w + 20.0
    np.testing.assert_allclose(joint, np.concatenate((alpha, line)), rtol=1e-12)
    np.testing.assert_allclose(sc.raw_counts(q, velocity, n, source, r_ratio, JN, pool, line_only=True), line,
                               rtol=1e-12)


@pytest.mark.slow
def test_the_drive_g_dtheta_and_f_lm_come_back_together(tmp_path):
    """Synthetic raw spectra of a triangular drive (512 channels, ±8 mm/s, at
    rest at +0.11 mm/s), alpha-Fe (T 8) and a saturated single line (T 15 at
    -0.1 mm/s), counted for different times: their own C_high and static noise.
    The source G 0.0765, Δθ 0.2, f_LM 0.73, d/D 0.25, Poisson at ~1e6 counts
    per channel; the procedure started from the usual source (G 0.09, Δθ 0.1)
    and f_LM 0.7. G, Δθ and f_LM come back within 3σ, the drive's step to 1e-4,
    and the files are written into the folder given."""
    n, JN, G, dtheta, f_lm, diameters = 512, 32, 0.0765, 0.2, 0.73, (5.0, 20.0)
    c_high, c_static = (3.0e4, 1.5e4), (2.0e3, 1.0e3)
    v = _triangle(n, 8.0, 0.11)
    pool = SerialPool()
    rng = np.random.default_rng(7)
    files = []
    for name, model, ns, absorber, ch, cs in (
            ("alpha_fe.mca", ['Sextet'], 7.0e5, [8.0, 0, 0, ALPHA_FE_FIELD, NAT_WIDTH, 0, 90, 0, 0, 0, 0, 0, 0, 3],
             c_high[0], c_static[0]),
            ("single_line.mca", ['Singlet'], 3.5e5, [15.0, -0.1, NAT_WIDTH, 0], c_high[1], c_static[1])):
        p = [ns, 0, 0, 0, ns * (1 - f_lm) / f_lm + ch, 0, 0, 0] + absorber
        files.append(_write_mca(tmp_path / name,
                                rng.poisson(sc.counts(v, p, model, None, G, dtheta, 0.25, JN, pool) + cs)))
    source = {'f_LM': 0.7, 'C_high': c_high, 'C_static': c_static}

    result = sc.calibrate_source(str(tmp_path), files, pool, JN, 1, 0.0, 0.0, diameters, source)

    g_error, dtheta_error, f_error = result['errors']
    assert abs(result['G'] - G) < 3 * g_error
    assert abs(result['dtheta'] - dtheta) < 3 * dtheta_error
    assert abs(result['f_LM'] - f_lm) < 3 * f_error
    assert result['start'] == (sc.START_G, sc.START_DTHETA)
    assert result['drive'][2] == pytest.approx(2 * 8.0 / (n // 2), abs=1e-4)
    assert result['hi2'] < 1.2
    assert os.path.exists(tmp_path / "Calibration.dat")
    stored = np.genfromtxt(tmp_path / "GCMS.txt")
    np.testing.assert_allclose(stored, [result['G'], result['dtheta'], *diameters])
