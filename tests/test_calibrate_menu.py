"""The Calibrate menu: Start calibration, and what the calibration's final fit
adds to the alpha-Fe sextet -- the impurity sextet (ON/OFF) and one of the
optics' doublets (Be / KB / none). CMS/SMS sets the mode's defaults (no
impurity sextet; Be for SMS, nothing for CMS), and a choice reaches the
calibration. What Calibration does with it: test_calibration_helpers.py
(the final model) and test_calibration.py (whole calibrations)."""
import os

import numpy as np

import syncmoss.syncmoss_main as sm

_SMS_MCA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alpha_fe_sms_000.mca")


def _entries(window):
    return {action.text(): action for action in window.cal_btn.menu().actions() if action.text()}


def _shown(window):
    """(impurity sextet, optics) as the menu shows them."""
    entries = _entries(window)
    impurity = "Impurity sextet (ON)" in entries
    assert impurity or "Impurity sextet (OFF)" in entries
    optics = [key for key, action in window.optics_actions.items() if action.isChecked()]
    assert len(optics) == 1, optics
    return impurity, optics[0]


def _chosen(window):
    return window.calibration_impurity, window.calibration_optics


def test_the_choices_and_the_modes_defaults(physics_app):
    w = physics_app
    assert w.cal_btn.menu().actions()[0].text() == "Start calibration"
    assert _chosen(w) == _shown(w) == (False, 'Be')                  # SMS, at the start

    _entries(w)["Impurity sextet (OFF)"].trigger()
    _entries(w)["Contain KB"].trigger()
    assert _chosen(w) == _shown(w) == (True, 'KB')
    _entries(w)["No impurity from optics"].trigger()                  # one at a time
    assert _chosen(w) == _shown(w) == (True, 'none')

    # switching the mode sets its defaults, whatever was chosen
    w.MS_fit.setChecked(True)
    assert _chosen(w) == _shown(w) == (False, 'none')
    _entries(w)["Impurity sextet (OFF)"].trigger()
    _entries(w)["Contain Be"].trigger()
    w.SMS_fit.setChecked(True)
    assert _chosen(w) == _shown(w) == (False, 'Be')


def test_start_calibration_passes_the_choice_on(physics_app, monkeypatch):
    w = physics_app
    _entries(w)["Impurity sextet (OFF)"].trigger()
    _entries(w)["Contain KB"].trigger()
    w.path_list = [_SMS_MCA]
    started = []
    monkeypatch.setattr(w, '_start_calculation', started.append)
    _entries(w)["Start calibration"].trigger()
    assert len(started) == 1 and started[0].impurities == (True, 'KB')

    received = {}

    def calibration(*args, **kwargs):
        received.update(kwargs)
        v = np.linspace(-5.0, 5.0, 16)
        return v, np.full(16, 100.0), np.full(16, 99.0)

    monkeypatch.setattr(sm, 'Calibration', calibration)
    started[0].run()                       # here: Calibration, then the window's handler
    assert received['impurities'] == (True, 'KB')
    assert "Calibration completed" in w.log.toPlainText()
    assert not w.inprogress
