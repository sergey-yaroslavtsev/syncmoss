"""Find and Refine for the SIMULATED instrumental function.

The conventional search is covered end to end by syncmoss_test.py
(``instrumental_pressed(1, 2)``). This is the equivalent for the simulated
57FeBO3 shape, for both buttons:

    Find   (ref 2)  discards what is stored and starts from the built-in values
    Refine (ref 3)  continues from the shape currently in INSth.txt

It runs on ``parameters/Calibration.dat``, which ships with the package. No
test may depend on the SL_studies series, which is outside the repository.

A smaller synthetic spectrum was tried and dropped: 128 points against 511 made
the tests only ~5 % faster, because the cost is the search's model evaluations
and not the length of the data. Not worth a second fixture to maintain.

SPEED. The full schedule is four annealing passes and takes a minute or two;
that is not what these tests are checking, so the schedule is cut to a single
short pass. What is being checked is the PLUMBING -- that each button starts
where it should, stores what it should, and leaves the other description alone
-- and that is independent of how far the minimiser is allowed to run. The
passes are shortened rather than the physics faked, so the real code path runs.
"""
import os
import shutil

import numpy as np
import pytest

from syncmoss import instrumental_io as iio
import syncmoss.sms_theory as smst

from conftest import redirect_params_dir_to_tmp


STORED_THETA = 55.0          # distinctive, so "did Refine continue?" is visible


@pytest.fixture
def quick_search(monkeypatch):
    """One short pass instead of the annealing schedule.

    'shift' alone is deliberate: with the shift factored out of the shape cache
    it costs no dynamical-diffraction rebuild, so the pass is fast, and it also
    keeps the B_s scan out of the way (the scan only fires when B_s is
    released), which is tested separately.
    """
    monkeypatch.setattr(iio, "THEORY_PASSES", (('shift',),))
    monkeypatch.setattr(iio, "THEORY_PASS_MI", 2)
    monkeypatch.setattr(iio, "THEORY_ESCALATION_CHI2", 1e9)   # never escalate


@pytest.fixture
def app(physics_app, tmp_path, quick_search):
    redirect_params_dir_to_tmp(physics_app, tmp_path)
    physics_app.SMS_fit.setChecked(True)
    physics_app.MS_fit.setChecked(False)
    physics_app.jn0_input.setText("16")
    cal = os.path.join(physics_app.params_dir, "Calibration.dat")
    physics_app.calibration_path = cal
    physics_app.process_path.setPlainText(repr([cal]))
    physics_app.path_list = [cal]
    assert physics_app.initialize_parameters()
    return physics_app


def _pool():
    from multiprocessing.pool import ThreadPool
    return ThreadPool(processes=2)


def _run(app, ref):
    pool = _pool()
    try:
        return iio.instrumental_theory(app, ref=ref, mode=2, pool=pool)
    finally:
        pool.close()
        pool.join()


def _store_a_distinctive_shape(app):
    kw = dict(iio.THEORY_FIXED)
    kw.update(iio.THEORY_START)
    kw['theta_urad'] = STORED_THETA
    iio.write_accurate_instrumental(app, smst.encode_physical(**kw))


def test_find_stores_a_usable_physical_shape(app):
    result = _run(app, ref=2)
    INS = result['INS']                      # a DICT is returned, not the array
    assert smst.ins_kind(INS) == smst.KIND_PHYSICAL
    stored = iio.read_accurate_instrumental(app)
    assert stored is not None
    assert np.allclose(stored, INS)
    fwhm, centre, _area = smst.ins_metrics(INS)
    assert 0.5 * 0.098 < fwhm < 20 * 0.098
    assert abs(centre) < 2.0


def test_find_ignores_what_is_stored(app):
    """Find restarts from the built-in values -- that is what makes it Find."""
    _store_a_distinctive_shape(app)
    result = _run(app, ref=2)
    assert result['theory']['theta_urad'] == pytest.approx(
        iio.THEORY_START['theta_urad']), "Find continued from the stored shape"


def test_refine_continues_from_what_is_stored(app):
    """Refine must NOT fall back to the defaults.

    ``theory_start_from_app`` tested ``ref != 1`` while Theory-Refine is ref 3,
    so Refine restarted from the built-in values every time. theta is held by
    the schedule, so it survives the fit and shows which start was used.
    """
    _store_a_distinctive_shape(app)
    result = _run(app, ref=3)
    assert result['theory']['theta_urad'] == pytest.approx(STORED_THETA), \
        "Refine restarted from the defaults instead of continuing"


def test_refine_without_anything_stored_uses_the_defaults(app):
    iio.write_accurate_instrumental(app, None)
    result = _run(app, ref=3)
    assert result['theory']['theta_urad'] == pytest.approx(
        iio.THEORY_START['theta_urad'])


def test_the_theory_search_leaves_the_gaussians_alone(app):
    """It used to overwrite INSexp.txt with a derived Gaussian stand-in,
    destroying a description the user had fitted."""
    path = os.path.join(app.params_dir, iio.INS_EXP_FILE)
    before = open(path, 'rb').read()
    _run(app, ref=2)
    assert open(path, 'rb').read() == before, "INSexp.txt was overwritten"


def test_the_conventional_search_leaves_the_theory_alone():
    """The mirror case, and the one that actually bit.

    ``instrumental()`` used to end with ``write_accurate_instrumental(app,
    None)``, because presence of the file was once what made the theoretical
    shape current. With the description chosen by a SETTING, and the Gaussian
    sum the default, one conventional search silently destroyed INSth.txt and
    "Theory" then fell back to the Gaussians for ever.

    Checked on the source: running the conventional alpha-Fe search for real
    costs a minute, and what must never come back is one specific line.
    """
    import inspect
    src = inspect.getsource(iio.instrumental)
    assert "write_accurate_instrumental(app, None)" not in src, \
        "the conventional search deletes the theoretical instrumental function"
