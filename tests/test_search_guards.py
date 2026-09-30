"""Guards around the theoretical instrumental-function search.

Three things that are easy to break and only ever fail in front of a user:

* the FIND dialog. A theory "Find" throws away the stored shape and restarts
  from built-in values, so it confirms first and lets the user correct the
  numbers they may know better than the defaults do. B_s and shift are
  deliberately absent -- Find maps both before releasing either.
* the BUSY guard. One calculation at a time: they share the pool, the
  integration grid and the instrumental function. The guard existed but
  returned silently, so clicking Fit during a search looked like a dead button.
* the INTERRUPT flag. ``fit_cancel`` is a threading.Event, the same mechanism
  and the same name the SYNCtime branch uses, because terminating the pool does
  not stop this search -- it rebuilds the source shape in its own thread.

The dialogs are modal, so ``QDialog.exec`` is replaced throughout; calling them
for real would hang the suite.
"""
import threading

import pytest

from PySide6.QtWidgets import QDialog, QLineEdit

import syncmoss.supp_menu as supp_menu
from syncmoss.instrumental_io import THEORY_BOUNDS, THEORY_START


FIND_FIELDS = ('theta_urad', 'dEQ', 'f_LM')


def _drive(monkeypatch, accept=True, edits=None):
    """Fill the Find dialog's boxes in order, then accept or reject."""
    def fake_exec(self):
        editors = self.findChildren(QLineEdit)
        assert len(editors) == len(FIND_FIELDS), "one box per asked-for field"
        for i, text in (edits or {}).items():
            editors[i].setText(text)
        return (QDialog.DialogCode.Accepted if accept
                else QDialog.DialogCode.Rejected)
    monkeypatch.setattr(QDialog, "exec", fake_exec)
    warned = []
    monkeypatch.setattr(supp_menu.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[-1]))
    return warned


def test_find_dialog_is_prefilled_with_the_defaults(physics_app, monkeypatch):
    _drive(monkeypatch)
    out = supp_menu.open_theory_find_dialog(physics_app)
    assert out is not None
    assert set(out) == set(FIND_FIELDS)
    for k in FIND_FIELDS:
        assert out[k] == pytest.approx(float(THEORY_START[k]), abs=1e-4)


def test_find_dialog_does_not_ask_for_what_it_maps(physics_app, monkeypatch):
    """Find maps B_s AND shift over a grid before releasing either, so asking
    for them would be asking for the very numbers the map is there to find."""
    _drive(monkeypatch)
    out = supp_menu.open_theory_find_dialog(physics_app)
    assert 'B_s' not in out
    assert 'shift' not in out


def test_find_dialog_cancel_returns_none(physics_app, monkeypatch):
    _drive(monkeypatch, accept=False)
    assert supp_menu.open_theory_find_dialog(physics_app) is None


def test_find_dialog_accepts_an_edited_value(physics_app, monkeypatch):
    _drive(monkeypatch, edits={0: "0"})          # theta -> the rocking minimum
    out = supp_menu.open_theory_find_dialog(physics_app)
    assert out['theta_urad'] == pytest.approx(0.0)


@pytest.mark.parametrize("index, text", [
    (0, "not a number"),
    (0, "99999"),                                # outside the theta bounds
    (2, "0.01"),                                 # outside the f_LM bounds
])
def test_find_dialog_refuses_a_bad_value(physics_app, monkeypatch, index, text):
    warned = _drive(monkeypatch, edits={index: text})
    assert supp_menu.open_theory_find_dialog(physics_app) is None
    assert warned, "the user should have been told why"


def test_find_dialog_bounds_match_the_search(physics_app):
    """The dialog validates against the SAME bounds the search uses, so a value
    it accepts is never refused a moment later."""
    for k in FIND_FIELDS:
        assert k in THEORY_BOUNDS
        lo, hi = THEORY_BOUNDS[k]
        assert lo <= float(THEORY_START[k]) <= hi, k


def test_busy_guard_refuses_and_explains(physics_app):
    physics_app.inprogress = False
    assert physics_app._reject_if_busy('Fitting') is False

    physics_app.inprogress = True
    physics_app.busy_with = 'The instrumental-function search'
    try:
        assert physics_app._reject_if_busy('Fitting') is True
    finally:
        physics_app.inprogress = False


def test_every_calculation_entry_point_is_guarded(physics_app):
    """Fit, Show model, Show spectrum, Calibration and the search itself."""
    import inspect
    src = inspect.getsource(type(physics_app))
    for name in ('fit_pressed', 'showM_pressed', 'show_pressed',
                 'calibration', 'instrumental_pressed'):
        body = src.split(f"def {name}(")[1].split("\n    def ")[0]
        assert '_reject_if_busy(' in body, f"{name} is not guarded"


def test_interrupt_flag_is_an_event(physics_app):
    """Same mechanism and name as SYNCtime: an Event, because a thread cannot
    be killed. What interrupt() does with it: test_interrupt.py."""
    assert isinstance(physics_app.fit_cancel, threading.Event)
    assert not physics_app.fit_cancel.is_set()
