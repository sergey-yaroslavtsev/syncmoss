"""SMS and CMS each have their own transmission-integral point count.

There used to be ONE stored value plus a switch: entering CMS turned 32 into 64
and leaving it turned 64 back into 32 -- but only while the value was still
exactly one of those two, so any value the user had edited silently stopped
following the mode. The switch is gone. There are now two independent settings,
``jn0_sms_input`` (32) and ``jn0_cms_input`` (64), edited together in
Supp -> "Set number of points for full transmission integral".

``jn0_input`` still exists and still holds the ACTIVE count -- everything
downstream reads it -- and ``refresh_jn0_for_mode`` is what keeps it in step
with the CMS/SMS checkboxes.

Both dialogs here are modal, so ``QDialog.exec`` and ``QMessageBox.warning``
are replaced; calling them for real would hang the suite.
"""
import pytest

from PySide6.QtWidgets import QDialog, QLineEdit, QMessageBox

import syncmoss.supp_menu as supp_menu


def _set_mode(window, cms):
    """Check CMS or SMS through the real handler, as a click would."""
    box = window.MS_fit if cms else window.SMS_fit
    box.setChecked(True)
    window.on_ms_sms_changed(box)


def _drive(monkeypatch, sms, cms, accept=True):
    """Type into both boxes, then accept or reject the dialog."""
    def fake_exec(self):
        editors = self.findChildren(QLineEdit)
        assert len(editors) == 2, "expected an SMS box and a CMS box"
        editors[0].setText(sms)
        editors[1].setText(cms)
        return (QDialog.DialogCode.Accepted if accept
                else QDialog.DialogCode.Rejected)

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[-1]))
    return warned


def test_defaults_are_32_and_64(physics_app):
    assert physics_app.jn0_sms_input.text() == "32"
    assert physics_app.jn0_cms_input.text() == "64"
    # SMS is the startup mode, so the active value is the SMS one
    assert physics_app.SMS_fit.isChecked()
    assert physics_app.jn0_input.text() == "32"


def test_mode_selects_the_matching_value(physics_app):
    _set_mode(physics_app, cms=True)
    assert physics_app.jn0_input.text() == "64"
    _set_mode(physics_app, cms=False)
    assert physics_app.jn0_input.text() == "32"


def test_no_doubling_of_an_edited_value(physics_app):
    """The old switch multiplied 32 -> 64. An edited value must be left alone."""
    physics_app.jn0_sms_input.setText("40")
    physics_app.jn0_cms_input.setText("100")
    _set_mode(physics_app, cms=False)
    assert physics_app.jn0_input.text() == "40"
    _set_mode(physics_app, cms=True)
    assert physics_app.jn0_input.text() == "100"      # not 80, not 64
    _set_mode(physics_app, cms=False)
    assert physics_app.jn0_input.text() == "40"       # and back, unchanged


def test_dialog_stores_both_and_refreshes(physics_app, monkeypatch):
    _drive(monkeypatch, "40", "80")
    assert supp_menu.open_integral_points_dialog(physics_app) is True
    assert physics_app.jn0_sms_input.text() == "40"
    assert physics_app.jn0_cms_input.text() == "80"
    assert physics_app.jn0_input.text() == "40"       # SMS is active
    _set_mode(physics_app, cms=True)
    assert physics_app.jn0_input.text() == "80"


def test_cancel_changes_nothing(physics_app, monkeypatch):
    _drive(monkeypatch, "1", "2", accept=False)
    assert supp_menu.open_integral_points_dialog(physics_app) is False
    assert physics_app.jn0_sms_input.text() == "32"
    assert physics_app.jn0_cms_input.text() == "64"


@pytest.mark.parametrize("sms, cms", [
    ("abc", "80"),          # not a number
    ("40", "0"),            # below the minimum
    ("40", "2000000"),      # above the maximum
])
def test_a_bad_value_leaves_the_pair_untouched(physics_app, monkeypatch,
                                               sms, cms):
    """Validate both before storing either -- never half-apply the pair."""
    warned = _drive(monkeypatch, sms, cms)
    assert supp_menu.open_integral_points_dialog(physics_app) is False
    assert warned, "the user should have been told"
    assert physics_app.jn0_sms_input.text() == "32"
    assert physics_app.jn0_cms_input.text() == "64"
