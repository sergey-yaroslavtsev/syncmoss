"""The instrumental function is shown in its OWN window, not on the main canvas.

Supp -> "Plot instrumental function from memory / from spectrum" opens a separate
top-level viewer and reuses it on every later call, so it can stay open next to
the spectrum being worked on. These tests pin that (and the precedence of the
theoretical instrumental function over the conventional one, which is what the
viewer reports).
"""
import os

import numpy as np
import pytest

from syncmoss import instrumental_io as iio
from syncmoss import supp_menu as sm
from syncmoss import sms_theory as st
# NOT "from tests.conftest": in CI's wheel mode pytest runs from a temp
# directory against the installed package, so the repo root is not on sys.path
# and "tests" is not an importable package. pytest does put the test file's own
# directory there, so the bare module name works in both modes.
from conftest import redirect_calibration_to_tmp  # noqa: F401  (fixtures)

pytestmark = pytest.mark.gui


def _redirect_params(window, tmp_path):
    """Point params_dir at a throw-away copy so the tracked files stay clean."""
    import shutil
    dst = os.path.join(str(tmp_path), "parameters")
    shutil.copytree(window.params_dir, dst, dirs_exist_ok=True)
    window.params_dir = dst
    window.dir_path = str(tmp_path)
    return dst


def test_plot_from_memory_opens_a_separate_window(physics_app, tmp_path):
    _redirect_params(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    physics_app.MS_fit.setChecked(False)

    assert getattr(physics_app, 'instrumental_window', None) is None
    main_figure = physics_app.figure

    sm.plot_instrumental_from_memory(physics_app)

    win = getattr(physics_app, 'instrumental_window', None)
    assert win is not None, "no separate window was created"
    assert win is not physics_app
    assert win.figure is not main_figure, "it drew on the main figure"
    assert win.isVisible()
    assert len(win.figure.axes) >= 1
    assert "Instrumental function" in win.windowTitle()
    # the summary line names the shape that is actually in use
    assert "Gaussian" in win.summary.text()


def test_the_window_is_reused_not_recreated(physics_app, tmp_path):
    _redirect_params(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    physics_app.MS_fit.setChecked(False)

    sm.plot_instrumental_from_memory(physics_app)
    first = physics_app.instrumental_window
    sm.plot_instrumental_from_memory(physics_app)
    assert physics_app.instrumental_window is first


def test_the_plot_follows_the_selected_method(physics_app, tmp_path):
    """Which description is drawn is the user's SETTING, not which file exists.

    It used to be the latter -- presence of the stored theoretical shape made it
    "current" -- and the plot drew it together with a Gaussian stand-in for the
    same source. Both are gone: the setting decides, and ONE curve is drawn, the
    one in use, so a stand-in can never be mistaken for what the program is
    using.
    """
    _redirect_params(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    physics_app.MS_fit.setChecked(False)

    acc = st.encode_physical(theta_urad=70.0, B_s=0.9, dEQ=-0.18, shift=-0.2)
    iio.write_accurate_instrumental(physics_app, acc)

    physics_app.instrumental_method = 'theory'
    sm.plot_instrumental_from_memory(physics_app)
    win = physics_app.instrumental_window
    text = win.summary.text() + win.windowTitle()
    assert "Simulation of 57FeBO3" in text
    assert iio.INS_TH_FILE in text
    assert len(win.figure.axes[0].lines) == 1, "only the shape in use is drawn"

    # the SAME stored shape, but the Gaussians selected: they must be plotted
    physics_app.instrumental_method = 'gauss'
    sm.plot_instrumental_from_memory(physics_app)
    win = physics_app.instrumental_window
    assert "Gaussian" in win.summary.text() + win.windowTitle()
    assert len(win.figure.axes[0].lines) == 1


def test_theory_falls_back_when_nothing_is_stored(physics_app, tmp_path):
    """Selecting Theory with no INSth.txt yet must not fail -- it falls back to
    the Gaussians and the label says which one was actually drawn."""
    _redirect_params(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    physics_app.MS_fit.setChecked(False)
    iio.write_accurate_instrumental(physics_app, None)

    physics_app.instrumental_method = 'theory'
    sm.plot_instrumental_from_memory(physics_app)
    win = physics_app.instrumental_window
    assert "Gaussian" in win.summary.text() + win.windowTitle()
    assert len(win.figure.axes[0].lines) == 1


def test_cms_mode_reports_instead_of_plotting(physics_app, tmp_path):
    _redirect_params(physics_app, tmp_path)
    physics_app.MS_fit.setChecked(True)
    physics_app.SMS_fit.setChecked(False)

    sm.plot_instrumental_from_memory(physics_app)
    assert getattr(physics_app, 'instrumental_window', None) is None
