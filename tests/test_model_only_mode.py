"""The model-only ("simulation") path-box keyword and the shipped help docs.

Typing ``Model_<range>`` into the spectrum-path box asks for the model to be
calculated on a synthetic velocity grid, with no experimental spectrum. Anything
else in that box is a spectrum path. Two things are pinned here:

  * the keyword parser: which forms are accepted and which malformed ranges are
    reported;
  * that a path which is neither an existing file nor the keyword ('models_6',
    'spectrum.dat' with a typo, ...) is refused with ONE message naming both
    possibilities — and refused BEFORE any worker thread starts, which is the
    bug this file was written for: such a path used to reach the loader inside
    the show-model thread and take the application down.

Plus a check that both markdown documents reachable from the Supp menu are
shipped and resolvable, since they are data files that packaging can drop.
"""
import os

import pytest

from syncmoss.models_description_window import (
    _replace_latex_math_blocks, resolve_help_path, resolve_models_description_path,
)

pytestmark = pytest.mark.gui


def _parse(window, text):
    window.process_path.setPlainText(text)
    return window.parse_model_only_request()


# --- the keyword itself -----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Model_6", 6.0),
    ("model_6", 6.0),
    ("MODEL_15", 15.0),
    ("Model_6.5", 6.5),
    ("['Model_6']", 6.0),
])
def test_valid_model_only_keyword(physics_app, text, expected):
    is_model_only, velocity_range, error = _parse(physics_app, text)
    assert is_model_only is True
    assert velocity_range == pytest.approx(expected)
    assert error is None


@pytest.mark.parametrize("text", ["Model_abc", "Model_", "Model_0", "Model_-3"])
def test_malformed_range_is_reported_not_silently_accepted(physics_app, text):
    is_model_only, velocity_range, error = _parse(physics_app, text)
    assert is_model_only is True
    assert velocity_range is None
    assert error


@pytest.mark.parametrize("text", [
    "models_6",                         # the typo that used to crash the app
    "spectrum.dat",                     # ordinary name
    "model.dat",                        # starts with 'model' but is a file name
    "data/models_6",
    "['a.dat', 'b.dat']",
    "",
])
def test_anything_else_is_a_spectrum_path(physics_app, text):
    """Only the exact keyword is model-only; everything else is a path, and a
    bad one is caught by the existence check rather than by guessing at typos."""
    assert _parse(physics_app, text) == (False, None, None)


def test_existing_file_is_accepted_whatever_it_is_called(physics_app, tmp_path):
    """A file that really is called 'models_6' must still be treated as a file."""
    real = tmp_path / "models_6"
    real.write_text("-1 100\n0 90\n1 100\n", encoding="utf-8")
    assert physics_app.check_spectrum_paths_exist([str(real)]) is None


# --- one message for "neither a file nor the keyword" -----------------------

def _assert_path_error(window):
    text = window.log.toPlainText()
    assert "red" in window.log.styleSheet().lower(), text
    # The single diagnosis must name both ways out.
    assert "Not a spectrum file" in text and "Model_<range>" in text


@pytest.mark.parametrize("text", ["models_6", "not_a_real_file.dat"])
def test_show_model_refuses_a_path_that_is_not_a_file(physics_app, text):
    """The former crash path: the loader is never reached, so the worker thread
    that used to fail inside it is never started."""
    w = physics_app
    w.process_path.setPlainText(text)
    w.showM_pressed()
    assert w.inprogress is False
    _assert_path_error(w)
    assert getattr(w, 'show_model_thread', None) is None


@pytest.mark.parametrize("action", ["show_pressed", "fit_pressed"])
def test_show_spectrum_and_fit_refuse_it_too(physics_app, action):
    w = physics_app
    w.process_path.setPlainText("models_6")
    getattr(w, action)()
    assert w.inprogress is False
    _assert_path_error(w)


# --- the documents behind the Supp menu ------------------------------------

def test_supp_menu_offers_both_documents(physics_app):
    texts = [a.text() for a in physics_app.supp_menu.actions()]
    assert any('Models description' in t for t in texts)
    assert any('Help' in t for t in texts)


@pytest.mark.parametrize("opener,attribute", [
    ('open_models_description_pressed', 'models_description_window'),
    ('open_help_pressed', 'help_window'),
])
def test_document_viewers_open_independently(physics_app, opener, attribute):
    """Each document gets its own window, so both can be open at once, and a
    second call reuses it instead of stacking windows.

    The window is destroyed again before the test ends: a stray top-level widget
    left alive head-less is what makes this suite hang at interpreter shutdown
    (see the force-exit note in conftest.py).
    """
    import syncmoss.supp_menu as supp_menu
    from PySide6.QtWidgets import QApplication

    w = physics_app
    assert getattr(w, attribute) is None
    try:
        getattr(supp_menu, opener)(w)
        window = getattr(w, attribute)
        assert window is not None
        assert window.viewer.toPlainText().strip()
        getattr(supp_menu, opener)(w)
        assert getattr(w, attribute) is window
    finally:
        window = getattr(w, attribute)
        setattr(w, attribute, None)
        if window is not None:
            window.close()
            window.setParent(None)
            window.deleteLater()
        QApplication.processEvents()


@pytest.mark.parametrize("resolve", [resolve_models_description_path, resolve_help_path])
def test_supp_menu_documents_are_shipped_and_render(physics_app, resolve):
    path = resolve(physics_app.dir_path)
    assert os.path.isfile(path), f"missing shipped document: {path}"
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    rendered = _replace_latex_math_blocks(text)
    assert rendered.strip()
    # No LaTeX math delimiter may survive the conversion: a leftover '$' means a
    # span the viewer could not read (e.g. one wrapped over a line break).
    assert '$' not in rendered


def test_models_description_does_not_document_unreachable_models(physics_app):
    """Only models offered in the dropdown may have a section (bar the two
    explicitly-labelled deprecated ones)."""
    from syncmoss.parameters_table import MODEL_OPTIONS

    path = resolve_models_description_path(physics_app.dir_path)
    with open(path, encoding="utf-8") as fh:
        headings = [ln[4:].strip() for ln in fh if ln.startswith('### ')]

    allowed = set(MODEL_OPTIONS) | {'baseline', 'Nbaseline'}
    for heading in headings:
        if 'deprecated' in heading.lower():
            continue
        names = [n.strip() for n in heading.split(',')]
        assert any(n in allowed for n in names), \
            f"'{heading}' is documented but not selectable in the GUI"
