"""How the instrumental function is approximated is a SETTING, not a menu entry.

The theoretical (simulated 57FeBO3) search used to be reachable only through a
parallel set of "Instr. func. NEW" entries in both dropdowns -- six menu items
for what is one choice. It is now one setting, ``PhysicsApp.instrumental_method``,
edited from Supp -> "Choose how to approximate instrumental function", and the
Find/Refine entries dispatch on it.

What is pinned here:

* the default is the shipped Gaussian sum, from ONE constant;
* the ref codes the menus produce (0/1 Gaussians, 2/3 theory);
* neither dropdown advertises "NEW" any more;
* the Supp entry exists and the dialog stores the choice, including that
  Cancel leaves the setting alone.

The dialog is driven by replacing ``QDialog.exec``: it is modal, so a test that
called it for real would hang.
"""
import pytest

from PySide6.QtWidgets import QDialog, QLineEdit, QPushButton

import syncmoss.supp_menu as supp_menu


SUPP_ENTRY = "Choose how to approximate instrumental function"


def _menu_labels(menu):
    return [a.text().replace('\n', ' ') for a in menu.actions() if a.text()]


def test_default_method_is_the_shipped_one(physics_app):
    """The Gaussian sum is the default for now: the simulated shape is opt-in
    while it settles. One constant decides it, so the window and the loader can
    never disagree."""
    from syncmoss.instrumental_io import DEFAULT_INSTRUMENTAL_METHOD
    assert DEFAULT_INSTRUMENTAL_METHOD == 'gauss'
    assert physics_app.instrumental_method == DEFAULT_INSTRUMENTAL_METHOD


@pytest.mark.parametrize("method, find_ref, refine_ref", [
    ('theory', 2, 3),
    ('gauss', 0, 1),
])
def test_ref_codes_follow_the_setting(physics_app, method, find_ref, refine_ref):
    physics_app.instrumental_method = method
    assert physics_app.instrumental_ref(False) == find_ref
    assert physics_app.instrumental_ref(True) == refine_ref


def test_dropdowns_no_longer_offer_NEW(physics_app):
    labels = (_menu_labels(physics_app.instrumental_menu)
              + _menu_labels(physics_app.instrumental_menu2))
    assert labels, "the instrumental menus are empty"
    assert not [t for t in labels if 'NEW' in t]
    # the three Find entries and the three Refine entries are still there
    for stem in ('single line', 'pure a-Fe', 'model'):
        assert any(stem in t and t.startswith('Find') for t in labels), stem
        assert any(stem in t and t.startswith('Refine') for t in labels), stem


def test_supp_menu_has_the_entry(physics_app):
    menu = supp_menu.build_supp_menu(physics_app)
    labels = [a.text() for a in menu.actions() if a.text()]
    assert SUPP_ENTRY in labels
    # the Gaussian count moved INTO that dialog, so it is no longer its own entry
    assert not [t for t in labels if t.startswith("Set number of lines")]


def test_gaussian_count_is_edited_in_the_dialog(physics_app, monkeypatch):
    """The count is stored whichever method button is pressed."""
    physics_app.instrumental_number.setText("3")
    _drive_dialog(monkeypatch, 'gauss', n_gauss="7")
    supp_menu.open_instrumental_method_dialog(physics_app)
    assert physics_app.instrumental_method == 'gauss'
    assert physics_app.instrumental_number.text() == "7"
    # ... including when Theory is chosen: it belongs to the description, not
    # to the act of selecting it
    _drive_dialog(monkeypatch, 'theory', n_gauss="5")
    supp_menu.open_instrumental_method_dialog(physics_app)
    assert physics_app.instrumental_method == 'theory'
    assert physics_app.instrumental_number.text() == "5"


def test_cancel_does_not_store_the_count(physics_app, monkeypatch):
    physics_app.instrumental_number.setText("3")
    monkeypatch.setattr(QDialog, "exec",
                        lambda self: QDialog.DialogCode.Rejected)
    supp_menu.open_instrumental_method_dialog(physics_app)
    assert physics_app.instrumental_number.text() == "3"


def _drive_dialog(monkeypatch, key, n_gauss=None):
    """Click the button for method ``key``, then report Accepted.

    Selected BY LABEL, not by position: the buttons sit in different layouts
    (the Gaussian one shares a row with its count box) and ``findChildren``
    does not promise creation order across nested layouts.

    ``n_gauss`` first types that text into the Gaussian-count box, which is the
    only QLineEdit in the dialog.
    """
    title = dict((k, t) for k, t, _b in supp_menu.INSTRUMENTAL_METHODS)[key]

    def fake_exec(self):
        if n_gauss is not None:
            editors = self.findChildren(QLineEdit)
            assert len(editors) == 1, "expected exactly one count box"
            editors[0].setText(n_gauss)
        match = [b for b in self.findChildren(QPushButton)
                 if b.text().startswith(title)]
        assert len(match) == 1, f"expected one button labelled {title!r}"
        match[0].click()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)


def test_dialog_selects_gaussians(physics_app, monkeypatch):
    _drive_dialog(monkeypatch, 'gauss')
    assert supp_menu.open_instrumental_method_dialog(physics_app) == 'gauss'
    assert physics_app.instrumental_method == 'gauss'
    assert physics_app.instrumental_ref(False) == 0


def test_dialog_selects_theory(physics_app, monkeypatch):
    physics_app.instrumental_method = 'gauss'
    _drive_dialog(monkeypatch, 'theory')
    assert supp_menu.open_instrumental_method_dialog(physics_app) == 'theory'
    assert physics_app.instrumental_method == 'theory'
    assert physics_app.instrumental_ref(True) == 3


def test_cancel_leaves_the_setting_alone(physics_app, monkeypatch):
    physics_app.instrumental_method = 'gauss'
    monkeypatch.setattr(QDialog, "exec",
                        lambda self: QDialog.DialogCode.Rejected)
    supp_menu.open_instrumental_method_dialog(physics_app)
    assert physics_app.instrumental_method == 'gauss'


def test_choice_is_independent_of_the_dat_file_toggle(physics_app):
    """The two settings are orthogonal and must stay that way.

    ``instrumental_method`` chooses how a SEARCH describes the source;
    ``use_dat_instrumental_metadata`` decides whether a spectrum that carries
    its own instrumental function uses it. Changing one must not move the other.
    """
    physics_app.use_dat_instrumental_metadata = True
    physics_app.instrumental_method = 'gauss'
    assert physics_app.use_dat_instrumental_metadata is True
    physics_app.instrumental_method = 'theory'
    assert physics_app.use_dat_instrumental_metadata is True
