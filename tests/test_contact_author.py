"""The highlighted "Contact the author" entry at the end of the Supp menu.

The entry cannot be colored through the palette (Qt has no per-action text
color), so the highlight is a bold font plus a painted envelope icon — and the
icon has to be re-painted on every theme switch, otherwise the accent chosen
for one background sits nearly invisible on the other.

It opens a small window, NOT a ``mailto:`` URL — that would hand the message to
whatever mail client the machine has registered, which is not where these users
write from.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from syncmoss import supp_menu
from syncmoss.constants import AUTHOR_EMAIL, ISSUES_URL


def test_accent_differs_between_the_two_modes():
    assert supp_menu.contact_accent_color(True) != supp_menu.contact_accent_color(False)


def test_no_mail_client_is_ever_launched():
    """QDesktopServices is the only way out to the machine's mail client.

    Re-importing it here is what a re-introduced ``mailto:`` would need, and
    handing the message to whatever client is registered (Outlook, as a rule)
    is exactly what this entry must not do — the address is shown and copied
    instead.
    """
    assert not hasattr(supp_menu, 'QDesktopServices')


@pytest.mark.gui
def test_contact_entry_is_last_and_highlighted(physics_app):
    w = physics_app
    actions = w.supp_menu.actions()

    assert actions[-1] is w.contact_action
    assert w.contact_action.text() == "Contact the author"
    assert actions[-2].isSeparator()            # set apart from the rest
    assert w.contact_action.font().bold()
    assert not w.contact_action.icon().isNull()


@pytest.mark.gui
def test_contact_icon_follows_the_theme(physics_app):
    w = physics_app
    before = w.contact_action.icon().cacheKey()
    w.toggle_theme()
    try:
        assert w.contact_action.icon().cacheKey() != before
    finally:
        w.toggle_theme()


@pytest.mark.gui
def test_contact_dialog_shows_the_address_and_the_issue_tracker(qapp):
    dialog = supp_menu.ContactDialog()
    try:
        assert dialog.email_label.text() == AUTHOR_EMAIL
        # selectable, so the address can be copied by hand as well
        assert dialog.email_label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse

        labels = [w.text() for w in dialog.findChildren(QLabel)]
        assert any(ISSUES_URL in text for text in labels)
        assert any('.mdl' in text for text in labels)   # what to attach

        assert dialog.copy_email() is True
        assert qapp.clipboard().text() == AUTHOR_EMAIL
        assert 'clipboard' in dialog.hint.text().lower()
    finally:
        dialog.deleteLater()


@pytest.mark.gui
def test_pressing_the_entry_reports_the_address(physics_app, monkeypatch):
    """The handler must not block the test on a modal window."""
    opened = []
    monkeypatch.setattr(supp_menu.ContactDialog, 'exec',
                        lambda self: opened.append(self))

    supp_menu.contact_author_pressed(physics_app)

    assert len(opened) == 1
    assert AUTHOR_EMAIL in physics_app.log.toPlainText()
