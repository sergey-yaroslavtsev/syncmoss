"""The Find box of the document viewers and of the License window (FindBar).

Pinned: matches are found ignoring case and highlighted, the current one in its
own colour; Enter / Shift+Enter walk them and wrap round; the current match
stays while the word is typed on; a new search starts at the text cursor; Esc
empties the box; a new text (a reloaded document, another license section) is
searched again. No main window is needed: the viewers are built on their own.
"""
import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest

from syncmoss.license_window import LicenseWindow
from syncmoss.models_description_window import (
    FIND_CURRENT_COLOR, FIND_MATCH_COLOR, ModelsDescriptionWindow,
)

pytestmark = pytest.mark.gui

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENT = "# Title\n\nalpha beta Alpha\n\ngamma ALPHA delta\n"


def _destroy(window):
    window.close()
    window.setParent(None)
    import shiboken6
    shiboken6.delete(window)


@pytest.fixture
def viewer(qapp, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(DOCUMENT, encoding="utf-8")
    window = ModelsDescriptionWindow(str(path), title="find test")
    try:
        yield window
    finally:
        _destroy(window)


def _found(bar):
    """The text of every match, read back from the document."""
    found = []
    for start, end in bar.matches:
        cursor = QTextCursor(bar.view.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        found.append(cursor.selectedText())
    return found


def _colors(bar):
    return [s.format.background().color().name() for s in bar.view.extraSelections()]


def test_matches_are_highlighted_and_walked_round(viewer):
    bar = viewer.find_bar
    bar.input.setText("alpha")
    assert _found(bar) == ["alpha", "Alpha", "ALPHA"]
    assert len(bar.matches) == viewer.viewer.toPlainText().lower().count("alpha")
    # from the top: the first one is current
    assert bar.count_label.text() == "1 / 3"
    assert _colors(bar) == [FIND_CURRENT_COLOR, FIND_MATCH_COLOR, FIND_MATCH_COLOR]

    for expected in (2, 3, 1):                    # Enter, wrapping round
        QTest.keyClick(bar.input, Qt.Key.Key_Return)
        assert bar.count_label.text() == f"{expected} / 3"
    QTest.keyClick(bar.input, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert bar.count_label.text() == "3 / 3"      # Shift+Enter, wrapping back
    assert _colors(bar) == [FIND_MATCH_COLOR, FIND_MATCH_COLOR, FIND_CURRENT_COLOR]
    assert viewer.viewer.textCursor().position() == bar.matches[2][1]
    assert bar.input.text() == "alpha"            # Enter never edits the box


def test_typing_on_keeps_the_current_match(viewer):
    bar = viewer.find_bar
    bar.input.setText("alpha")
    bar.next()
    bar.next()                                    # 'ALPHA' of 'gamma ALPHA delta'
    QTest.keyClick(bar.input, Qt.Key.Key_Space)   # 'alpha ': the line-end 'Alpha' drops out
    assert _found(bar) == ["alpha ", "ALPHA "]
    assert bar.count_label.text() == "2 / 2"


def test_a_new_search_starts_at_the_text_cursor(viewer):
    text = viewer.viewer
    cursor = text.textCursor()
    cursor.setPosition(text.document().find("gamma").selectionStart())
    text.setTextCursor(cursor)                    # as a click there does
    bar = viewer.find_bar
    bar.input.setText("alpha")
    assert bar.count_label.text() == "3 / 3"


def test_no_match_and_esc(viewer):
    bar = viewer.find_bar
    bar.input.setText("omega")
    assert bar.matches == [] and bar.current == -1
    assert bar.count_label.text() == "not found" and _colors(bar) == []
    bar.input.setText("beta")
    assert _colors(bar) == [FIND_CURRENT_COLOR]
    QTest.keyClick(bar.input, Qt.Key.Key_Escape)
    assert bar.input.text() == "" and bar.count_label.text() == "" and _colors(bar) == []


def test_a_reloaded_document_is_searched_again(viewer, tmp_path):
    bar = viewer.find_bar
    bar.input.setText("alpha")
    (tmp_path / "doc.md").write_text("one alpha only\n", encoding="utf-8")
    viewer.reload_document()
    assert _found(bar) == ["alpha"] and bar.count_label.text() == "1 / 1"


def test_license_window_finds_in_the_section_shown(qapp):
    # dir_path as PhysicsApp sets it when not frozen: the package folder
    window = LicenseWindow(os.path.join(_REPO, "syncmoss"))
    try:
        bar = window.find_bar
        bar.input.setText("license")
        counts = []
        for row in (0, 1):                        # the MIT license, the notices
            window.index.setCurrentRow(row)
            expected = window.viewer.toPlainText().lower().count("license")
            assert len(bar.matches) == expected > 0
            assert bar.count_label.text() == f"1 / {expected}"
            assert len(_colors(bar)) == expected
            counts.append(expected)
        assert counts[0] != counts[1]             # so the second text was searched anew
    finally:
        _destroy(window)
