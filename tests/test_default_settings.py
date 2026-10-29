"""Supp -> "Save settings as default" and "Restore default settings".

Save writes the window's switchable settings (PhysicsApp.current_settings) to
config.json in params_dir; main.py applies that file to the window it builds
(supp_menu.apply_saved_settings), so the next start has them; Restore deletes
the file. A window built here never reads it by itself -- only main.py does --
so the tests' windows have the built-in settings whatever the user saved.
"""
import json
import os

from conftest import _destroy_window
from syncmoss.supp_menu import (
    SETTINGS_FILE, apply_saved_settings, channel_average_text, chi2_correction_text, restore_default_settings,
    save_settings_as_default, toggle_channel_average, toggle_chi2_correction,
)
from syncmoss.syncmoss_main import ONE_MODEL, PhysicsApp

# One value of each kind a setting cannot take; the window must keep its own.
_NOT_VALID = {
    'cms': 'yes',                     # a switch that is not true/false
    'velocity_up_down': 1,
    'jn0_sms': 0,                     # below the store's range
    'jn0_cms': 64.0,                  # not whole
    'instrumental_number': 1001,      # above the store's range
    'polarization': 1.5,
    'sequence_fitting_type': 7,       # no such Multispectra mode
    'instrumental_method': 'magic',
    'channel_average': 1,             # a switch that is not true/false
    'calibration_impurity': 'on',
    'calibration_optics': 'Fe',       # no such optics impurity
}


def _path(window):
    return os.path.join(window.params_dir, SETTINGS_FILE)


def _change_every_setting(w):
    """Every setting off its built-in value, each through its own control."""
    w.toggle_theme()
    w.MS_fit.setChecked(True)                    # on_ms_sms_changed unchecks SMS
    w.choose_calibration_impurities(True, 'KB')  # what the Calibrate menu sets, after CMS
    w.toggle_velocity_direction()
    w.toggle_use_dat_instrumental()
    w.set_sequence_fitting_type(ONE_MODEL)
    w.instrumental_method = 'theory'             # what the method dialog sets
    w.instrumental_number.setText("5")
    toggle_chi2_correction(w, w.chi2_correction_action)
    toggle_channel_average(w, w.channel_average_action)
    w.jn0_sms_input.setText("40")                # what the integral-points dialog sets
    w.jn0_cms_input.setText("100")
    w.refresh_jn0_for_mode()
    w.polarization_input.setText("0.5")


def test_a_new_start_has_the_saved_settings(physics_app):
    w = physics_app
    built_in = w.current_settings()
    _change_every_setting(w)
    saved = w.current_settings()
    assert [key for key in built_in if saved[key] == built_in[key]] == []
    assert save_settings_as_default(w)
    with open(_path(w), encoding='utf-8') as f:
        assert json.load(f) == saved

    # the next start: a new window has the built-in settings, then main.py
    # applies the saved ones
    w._force_color_scheme(built_in['dark_mode'])
    new = PhysicsApp(pool=None)
    try:
        assert new.current_settings() == built_in
        new.params_dir = w.params_dir
        assert apply_saved_settings(new)
        assert new.current_settings() == saved
        # and shows them as the window that saved them does
        assert not new.SMS_fit.isChecked()
        assert new.SMS_pol == 0.5
        for name in ('jn0_input', 'instrumental_btn', 'seq_fit_btn', 'cal_cho_title',
                     'theme_action', 'chi2_correction_action', 'channel_average_action',
                     'impurity_sextet_action'):
            assert getattr(new, name).text() == getattr(w, name).text(), name
        assert new.optics_actions['KB'].isChecked() and not new.optics_actions['none'].isChecked()
    finally:
        _destroy_window(new)
        w._force_color_scheme(built_in['dark_mode'])


def test_restore_deletes_the_saved_settings_and_the_window_keeps_its_own(physics_app):
    w = physics_app
    entries = {action.text(): action for action in w.supp_menu.actions()}
    w.toggle_velocity_direction()
    entries["Save settings as default"].trigger()
    assert os.path.isfile(_path(w))
    kept = w.current_settings()

    entries["Restore default settings"].trigger()
    assert not os.path.exists(_path(w))
    assert w.current_settings() == kept
    assert not restore_default_settings(w)       # nothing left to delete
    assert not apply_saved_settings(w)           # a start without the file
    assert w.current_settings() == kept


def test_a_saved_setting_not_valid_keeps_the_windows_value(physics_app):
    w = physics_app
    built_in = w.current_settings()
    with open(_path(w), 'w', encoding='utf-8') as f:
        json.dump(dict(_NOT_VALID, chi2_correction=False, channel_average=False, a_later_setting=1), f)
    assert apply_saved_settings(w)
    # the valid ones are applied, the others keep their values
    assert w.current_settings() == dict(built_in, chi2_correction=False, channel_average=False)
    assert w.chi2_correction_action.text() == chi2_correction_text(False)
    assert w.channel_average_action.text() == channel_average_text(False)
    assert SETTINGS_FILE in w.log.toPlainText()
    # each not valid one is named; an unknown one is ignored
    assert sorted(w.apply_settings(dict(_NOT_VALID, a_later_setting=1))) == sorted(_NOT_VALID)


def test_an_unreadable_settings_file_changes_nothing(physics_app):
    w = physics_app
    built_in = w.current_settings()
    for text in ('{"cms": true', '[true]'):
        with open(_path(w), 'w', encoding='utf-8') as f:
            f.write(text)
        assert not apply_saved_settings(w), text
        assert w.current_settings() == built_in, text
        assert SETTINGS_FILE in w.log.toPlainText(), text
