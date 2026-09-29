"""The chi-square spread sqrt(2/dof) shown next to the fitted chi-square.

A reduced chi-square of 1.3 is one sigma on 40 degrees of freedom and nine on
4000; the spread on the result image and in the ``[Fit]`` line is what lets a
user tell the two apart.
"""
import numpy as np
from matplotlib.figure import Figure

from syncmoss.fitting_io import chi2_spread
from syncmoss.spectrum_plotter import (
    _chi2_title, plot_fitting_result, plot_simultaneous_fitting_result)


def test_spread_is_sqrt_two_over_dof():
    assert chi2_spread(1024, 24) == np.sqrt(2.0 / 1000)
    # more free parameters than points: clamp to one degree of freedom
    assert chi2_spread(5, 9) == np.sqrt(2.0)


def test_title_carries_the_spread_only_when_given():
    assert _chi2_title(1.02345, 0.04472) == 'χ² = 1.023 ± 0.045'
    assert _chi2_title(1.02345) == 'χ² = 1.023'
    assert _chi2_title(1.02345, float('nan')) == 'χ² = 1.023'


def _spectrum():
    A = np.linspace(-10.0, 10.0, 64)
    B = np.full_like(A, 1000.0)
    p = np.zeros(8)
    p[0] = 1000.0
    return A, B, B - 1.0, p


def test_single_fit_image_shows_the_spread(tmp_path):
    A, B, SPC_f, p = _spectrum()
    figure = Figure()
    # z_order given -> replot mode, nothing is written to dir_path
    plot_fitting_result(figure, A, B, SPC_f, [], [], p, ['r'], 1.02345,
                        str(tmp_path / 'one.dat'), str(tmp_path), z_order=[],
                        chi2_spread=0.04472)
    assert figure.get_axes()[0].get_title() == 'χ² = 1.023 ± 0.045'


def test_simultaneous_fit_image_shows_the_spread(tmp_path):
    A, B, SPC_f, p = _spectrum()
    figure = Figure()
    plot_simultaneous_fitting_result(
        figure, [A, A], [B, B], [SPC_f, SPC_f], [[], []], [[], []],
        np.concatenate([p, p]), [0, 8], ['r', 'g'], 1.02345,
        [str(tmp_path / 'a.dat'), str(tmp_path / 'b.dat')], str(tmp_path),
        z_order=[], chi2_spread=0.03162)
    titles = [ax.get_title() for ax in figure.get_axes() if ax.get_title()]
    assert titles == ['χ² = 1.023 ± 0.032']
