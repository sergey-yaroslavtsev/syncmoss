"""Generate a synthetic alpha-Fe calibration spectrum (.mca) for the calibration test.

A real velocity-calibration measurement is an alpha-Fe foil: a symmetric sextet
whose six lines sit at the well-known velocities ``sex0`` (mm/s). We reproduce a
CONVENTIONAL-source (CMS / MS-mode) run, whose drive ramps the velocity *linearly*
with a constant step and folds it (triangular sweep): channel -> velocity is two
mirror linear ramps, so each velocity is visited twice. Six Lorentzian dips at
``sex0`` (3:2:1:1:2:3 depths) plus Poisson noise give a realistic folded sextet.

(The SMS / synchrotron path uses a *sinusoidal* drive and is covered by the real
``alpha_fe_sms_000.mca`` fixture; MS mode uses the linear step generated here.)

Run this module to (re)write the committed ``.mca`` fixture. It is deterministic
(seeded), so regenerating gives byte-identical files.
"""
import os

import numpy as np

# alpha-Fe sextet line velocities (mm/s) -- the calibration reference points.
SEX0 = np.array([-5.3123, -3.0760, -0.8397, 0.8397, 3.0760, 5.3123])
# Relative line depths (3:2:1 for a random-powder / thin absorber).
DEPTHS = np.array([3.0, 2.0, 1.0, 1.0, 2.0, 3.0])


def _linear_drive(ch, n, vmax, step):
    """Channel -> velocity for a folded (triangular) linear drive.

    First half ramps down from ``vmax`` with a constant ``step`` per channel;
    the second half mirrors it. Matches the two-ramp form of Calibration.lin_cal.
    """
    v = np.empty(n)
    half = n // 2
    first = np.arange(half)
    v[:half] = vmax - step * first
    second = np.arange(half, n)
    v[half:] = vmax - step * (n - 1 - second)
    return v


def synth_spectrum(n=256, vmax=6.0, step=None, baseline=20000.0,
                   hwhm=0.16, outer_depth_frac=0.33, seed=2024):
    if step is None:
        # cover a bit beyond the outer alpha-Fe lines (+/-5.31 mm/s) over a half sweep
        step = 2 * vmax / (n // 2)
    ch = np.arange(n)
    v = _linear_drive(ch, n, vmax, step)
    absorption = np.zeros(n)
    scale = outer_depth_frac / DEPTHS.max()
    for v0, d in zip(SEX0, DEPTHS):
        absorption += d * scale * hwhm ** 2 / ((v - v0) ** 2 + hwhm ** 2)
    counts = baseline * (1.0 - absorption)
    rng = np.random.default_rng(seed)
    return rng.poisson(np.clip(counts, 1.0, None)).astype(int)


def write_mca(path, counts, title):
    n = len(counts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"#S 1 {title}\n")
        f.write(f"#@CHANN {n} 0 {n - 1} 1\n")
        # One long "@A" line of space-separated counts, terminated by a '#' line
        # (matches Calibration._load parsing of .mca).
        f.write("@A " + " ".join(str(int(c)) for c in counts) + "\n")
        f.write("#\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    counts = synth_spectrum()
    write_mca(os.path.join(here, "synthetic_alpha_fe_cms_linear.mca"), counts,
              "synthetic alpha-Fe CMS calibration (linear/triangular drive)")
    print("wrote synthetic_alpha_fe_cms_linear.mca")


if __name__ == "__main__":
    main()
