"""numba's compiled functions: their folder and its marker (jit_cache), its
locator in the frozen app (frozen_cache_locator), compiling them all
(compile_functions) in a new process, and the start that compiles them when the
folder is not complete for the running program (compile_window).

The frozen app itself is checked by its own --test (main.py): the compiling
must fill the folder, and the smoke flow after it must add nothing. Here the
frozen case is played by setting sys.frozen and a module file name relative
like PyInstaller's.
"""
import json
import os
import subprocess
import sys
import time
import types
from multiprocessing.pool import ThreadPool

import pytest
from numba import njit
from numba.core import config
from PySide6.QtCore import QCoreApplication

import syncmoss.compile_functions as cf
import syncmoss.compile_window as cw
import syncmoss.jit_cache as jit_cache
import syncmoss.models as m5
import syncmoss.syncmoss_main as sm
from syncmoss.frozen_cache_locator import FrozenCacheLocator
from syncmoss.models import FIT_CANCEL
from syncmoss.parameters_table import MODEL_OPTIONS

import compile_stand_ins as stand_ins
import syncmoss_test

TESTS = os.path.dirname(os.path.abspath(__file__))


def _pump_until(condition, timeout=120.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QCoreApplication.processEvents()
        time.sleep(0.01)


# ------------------------------------------------------------- the folder

@pytest.mark.parametrize("frozen_macos", [False, True])
def test_the_folder_is_next_to_the_parameters(monkeypatch, tmp_path, frozen_macos):
    # from source: the package folder, which holds parameters/
    assert jit_cache.cache_dir() == os.path.join(os.path.dirname(sm.__file__), "numba_cache")
    monkeypatch.setattr(jit_cache, "app_dir", lambda: str(tmp_path / "app"))
    monkeypatch.setattr(jit_cache, "is_frozen_macos", lambda: frozen_macos)
    monkeypatch.setattr(jit_cache, "macos_app_data_dir", lambda: str(tmp_path / "AppData"))
    # the frozen macOS app: in the per-user folder that IS its parameters folder
    assert jit_cache.cache_dir() == str(tmp_path / ("AppData" if frozen_macos else "app") / "numba_cache")


@pytest.mark.parametrize("frozen", [False, True])
def test_configure_points_numba_at_the_folder_before_numba_is_loaded(monkeypatch, tmp_path, frozen):
    monkeypatch.setattr(jit_cache, "cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_LOCATOR_CLASSES", raising=False)
    jit_cache.configure()                           # numba is loaded in this process: too late
    assert "NUMBA_CACHE_DIR" not in os.environ
    monkeypatch.delitem(sys.modules, "numba")
    jit_cache.configure()
    assert os.environ["NUMBA_CACHE_DIR"] == str(tmp_path)
    assert os.environ.get("NUMBA_CACHE_LOCATOR_CLASSES") == (jit_cache.FROZEN_LOCATORS if frozen else None)


def test_a_finished_compiling_is_marked_with_what_it_was_for(monkeypatch, tmp_path):
    folder = str(tmp_path / "SYNC[moss]" / "numba_cache")  # glob's own characters in the path
    current = jit_cache.stamp("0.7.1", "0.65.0")
    assert jit_cache.why_compile(folder, current) == "nothing compiled yet"
    os.makedirs(os.path.join(folder, "syncmoss"))           # what numba makes at the import
    assert jit_cache.why_compile(folder, current) == "nothing compiled yet"
    stand_ins.index(folder)                                 # a compiling stopped half way
    assert jit_cache.why_compile(folder, current) == "the last compiling did not finish"
    jit_cache.mark_compiled(folder, current)
    assert jit_cache.why_compile(folder, current) is None
    assert jit_cache.why_compile(folder, jit_cache.stamp("0.8.0", "0.65.0")) == \
        "compiled for SYNCmoss 0.7.1, this is 0.8.0"
    assert jit_cache.why_compile(folder, jit_cache.stamp("0.7.1", "0.66.0")) == \
        "compiled for numba 0.65.0, this is 0.66.0"
    # a frozen app: the executable, by which numba dates the functions there
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    frozen = jit_cache.stamp("0.7.1", "0.65.0")
    executable = os.stat(sys.executable)
    assert frozen["executable"] == [executable.st_mtime, executable.st_size]
    assert jit_cache.why_compile(folder, frozen) == "compiled by another executable"
    jit_cache.mark_compiled(folder, frozen)
    assert jit_cache.why_compile(folder, frozen) is None    # its time stamp read back exactly

    mine = os.path.join(folder, "syncmoss", "notes.txt")
    with open(mine, "w") as f:
        f.write("the user's")
    assert len(jit_cache.files(folder)) == 2
    jit_cache.clear(folder)
    assert jit_cache.why_compile(folder, frozen) == "nothing compiled yet"
    assert os.listdir(os.path.join(folder, "syncmoss")) == ["notes.txt"] and os.listdir(folder) == ["syncmoss"]
    jit_cache.mark_compiled(folder, frozen)
    assert jit_cache.why_compile(folder, frozen) == "no compiled function left"
    with open(os.path.join(folder, jit_cache.MARKER), "w") as f:
        f.write("{")
    assert jit_cache.why_compile(folder, frozen) == f"{jit_cache.MARKER} unreadable"


# ------------------------------------------------------ the frozen locator

def test_the_frozen_locator_keeps_the_functions_in_the_folder_wherever_the_program_starts(monkeypatch, tmp_path):
    folder = tmp_path / "numba_cache"
    monkeypatch.setattr(config, "CACHE_DIR", str(folder))
    relative = os.path.join("syncmoss", "models.py")        # PyInstaller's file name of a module

    def kernel():
        pass

    assert FrozenCacheLocator.from_function(kernel, relative) is None        # not frozen
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    paths = set()
    for start in ("a", "b"):                                 # Finder starts it from /
        (tmp_path / start).mkdir()
        monkeypatch.chdir(tmp_path / start)
        locator = FrozenCacheLocator.from_function(kernel, relative)
        paths.add(locator.get_cache_path())
    assert paths == {str(folder / "syncmoss")}
    # dated by the executable, numba's rule for a frozen app (its stamp's form
    # is numba's: (mtime, size) in 0.65, a hash of the file in 0.68)
    stamp = locator.get_source_stamp()
    assert locator.get_source_stamp() == stamp
    new_build = tmp_path / "new_build.exe"
    new_build.write_bytes(b"a new build")
    monkeypatch.setattr(sys, "executable", str(new_build))
    assert locator.get_source_stamp() != stamp
    assert FrozenCacheLocator.from_function(kernel, os.path.abspath(relative)) is None   # numba's own case
    monkeypatch.setattr(config, "CACHE_DIR", "")
    assert FrozenCacheLocator.from_function(kernel, relative) is None


def test_numba_takes_the_frozen_locator_by_the_name_configure_gives(monkeypatch, tmp_path):
    """numba resolves FROZEN_LOCATORS itself, stores a function compiled from a
    relative file name in the folder, and a new dispatcher (a new start) loads
    it from there."""
    config.reload_config()                          # numba's config as the environment is now; kept below
    folder = tmp_path / "numba_cache"
    monkeypatch.setattr(config, "CACHE_DIR", str(folder))
    monkeypatch.setattr(config, "CACHE_LOCATOR_CLASSES", jit_cache.FROZEN_LOCATORS)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.chdir(tmp_path)
    probe = types.ModuleType("probe_package.probe")         # a module, its file name PyInstaller's
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    exec(compile("def twice(x):\n    return 2.0 * x\n", os.path.join("probe_package", "probe.py"), "exec"),
         probe.__dict__)
    assert njit(cache=True)(probe.twice)(2.0) == 4.0
    assert jit_cache.compiled(str(folder)) and os.listdir(folder) == ["probe_package"]
    again = njit(cache=True)(probe.twice)
    assert again(3.0) == 6.0
    assert sum(again.stats.cache_hits.values()) == 1 and not again.stats.cache_misses


# ------------------------------------------------------------ the compiling

def test_the_compiling_has_a_row_of_every_fittable_model():
    fittable = MODEL_OPTIONS[:MODEL_OPTIONS.index('Be')]     # the dropdown's first part
    assert {name: len(row) for name, row in cf.ROWS.items()} == {name: m5._COMPONENT_SLOTS[name] for name in fittable}


def test_the_compiling_runs_and_counts_its_steps():
    steps = []
    cf.compile_functions(lambda *step: steps.append(step))
    names = list(cf.ROWS) + ["components search"]
    assert steps == [(done, len(names), name) for done, name in enumerate(names)]


# Cached kernels the program never calls, or only with models.COMPLEX_VOIGT_METHOD
# = 'off' (a developer's switch)
NOT_CALLED = {
    'syncmoss.models.' + name for name in (
        'sim_inv', 'sim_inv_3d', 'inv', 'D3A', 'MatMul', 'Lin_Mat3D_Mul', 'sol_t', 'Voigt_complex_easy',
        '_relax_MS_groups', 'relax_MS', 'Blume', 'ASM', 'K_cei', 'SN', 'relax_MS_thick')
} | {
    'syncmoss.models_positions.' + name for name in (
        'sim_inv', 'sim_inv_3d', 'inv', 'D2A', 'D3A', 'insert1D', 'MatMul', 'Lin_Mat3D_Mul', 'sol_t', 'Voigt',
        'relax_MS', 'Blume')
}

# In a new process with an empty folder: compile_functions, the kernels it
# leaves uncompiled, then every case of test_models_dual (value and
# derivatives, both sources; plain, and with the channel average and the CMS
# cosine smearing) and two Nbaseline sections of the two methods, and the
# compiled versions they added. (Loaded from a filled folder, a kernel does not
# list the kernels it calls as compiled: so an empty one.)
_REPORT = '''
import json
import sys
sys.path[:0] = sys.argv[1:3]
import numpy as np
from numba.core.caching import NullCache
from numba.core.registry import CPUDispatcher
import syncmoss.compile_functions as cf
import syncmoss.dual_kernels as dk
import syncmoss.dual_numbers as dn
import syncmoss.instrumental_io as iio
import syncmoss.minimi_lib as mi
import syncmoss.models as m5
import syncmoss.models_dual as md
import syncmoss.models_positions as mpos
import test_models_dual as tmd


def versions():
    return {f"{module.__name__}.{name}": {str(s) for s in kernel.signatures}
            for module in (m5, mpos, dk, mi) for name, kernel in vars(module).items()
            if isinstance(kernel, CPUDispatcher) and kernel.__module__ == module.__name__
            and not isinstance(kernel._cache, NullCache)}


def both(x, p, model, x0, MulCo, INS, Met, Norm, free, Distri=(0,), Cor=(0,), Recon=(0,), lengths=None, smearing=None):
    m5.TI(x, p, model, tmd.JN, tmd.SerialPool(), x0, MulCo, INS, list(Distri), list(Cor), Met=Met, Norm=Norm,
          pol=tmd.POL, Recon=list(Recon), lengths=lengths, smearing=smearing)
    prepared = md.prepare(x, model, tmd.JN, x0, MulCo, INS, Met, Norm, tmd.POL, lengths, smearing)
    md.TI(dn.seed(p, free), prepared, tmd.SerialPool(), list(Distri), list(Cor), list(Recon))


cf.compile_functions()
compiled = versions()
drive = m5.fit_drive(tmd.X)
for case in tmd.CASES.values():
    free = [k for k in range(len(case['p'])) if k not in case['fixed']]
    for s in tmd.SOURCES.values():
        smeared = dict(s, v_center=0.0, drive=drive, fold_offset=0.3, dtheta=0.19 * s['Met'], r_ratio=0.25)
        for Norm, smearing in ((s['Norm'], None), iio.source_for(tmd.SerialPool(), tmd.X, tmd.JN, smeared)):
            both(tmd.X, case['p'], case['model'], s['x0'], s['MulCo'], s['INS'], s['Met'], Norm, free,
                 case['Distri'], case['Cor'], case['Recon'], smearing=smearing)
sms, cms = tmd.SOURCES['SMS'], tmd.SOURCES['CMS']
p = np.array(tmd.BASELINE + tmd.SEXTET + tmd.BASELINE + tmd.DOUBLET, dtype=float)
both(np.concatenate([tmd.X, tmd.X]), p, ['Sextet', 'Nbaseline', 'Doublet'], [sms['x0'], cms['x0']],
     [sms['MulCo'], cms['MulCo']], [sms['INS'], cms['INS']], [0, 1], [sms['Norm'], cms['Norm']],
     list(range(len(p))), lengths=[len(tmd.X)] * 2)
added = {name: sorted(v - compiled[name]) for name, v in versions().items() if v - compiled[name]}
print(json.dumps({'uncompiled': sorted(name for name, v in compiled.items() if not v), 'added': added}))
'''


@pytest.mark.slow
def test_the_compiling_compiles_what_the_program_runs(tmp_path):
    script = tmp_path / "report.py"
    script.write_text(_REPORT, encoding="utf-8")
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"), PYTHONIOENCODING="utf-8")
    env.pop("NUMBA_CACHE_LOCATOR_CLASSES", None)
    run = subprocess.run([sys.executable, str(script), os.path.dirname(TESTS), TESTS], capture_output=True,
                         text=True, encoding="utf-8", env=env, timeout=900)
    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert set(report['uncompiled']) <= NOT_CALLED, sorted(set(report['uncompiled']) - NOT_CALLED)
    assert report['added'] == {}


def test_the_new_process_reports_its_steps_and_its_end(monkeypatch, tmp_path):
    monkeypatch.setenv(stand_ins.FOLDER_VARIABLE, str(tmp_path))
    steps = []
    assert jit_cache.run_in_new_process(stand_ins.leave_an_index, progress=lambda *step: steps.append(step))
    assert steps == [(0, 2, "probe"), (1, 2, "probe again")] and jit_cache.compiled(str(tmp_path))
    with pytest.raises(jit_cache.CompilingFailed, match="RuntimeError: no compiler today"):
        jit_cache.run_in_new_process(stand_ins.fail)
    started = time.monotonic()
    assert jit_cache.run_in_new_process(stand_ins.sleep_long, stop=lambda: time.monotonic() - started > 0.5) is False


# ----------------------------------------------------------------- the start

def test_a_start_compiles_unless_the_folder_is_complete_for_this_version(physics_app, monkeypatch, tmp_path, capsys):
    w = physics_app
    folder = tmp_path / "numba_cache"
    monkeypatch.setattr(w, "numba_cache_dir", str(folder))
    monkeypatch.setenv(stand_ins.FOLDER_VARIABLE, str(folder))
    FIT_CANCEL.clear()
    steps = []
    monkeypatch.setattr(cw.CompileDialog, "show_step", lambda dialog, *step: steps.append(step))
    stale = stand_ins.index(folder, "models.Old-1.py313")

    assert cw.compile_if_needed(w, work=stand_ins.leave_an_index)          # half done before: anew
    assert not os.path.exists(stale)
    assert w.inprogress and w.busy_with == cw.BUSY and w.compile_dialog.isVisible()
    assert w._reject_if_busy("Fit")                                      # the calculations wait
    _pump_until(lambda: not w.inprogress)
    assert steps == [(0, 2, "probe"), (1, 2, "probe again")]
    assert not w.compile_dialog.isVisible()
    assert w.log.toPlainText().startswith("Calculation functions compiled: ready")
    assert f"{cw.BUSY} took" in w.log.toPlainText()
    assert jit_cache.why_compile(str(folder), cw.current_stamp()) is None
    assert "the last compiling did not finish" in capsys.readouterr().out

    assert not cw.compile_if_needed(w, work=stand_ins.fail)              # complete: nothing to do
    assert cw.compile_if_needed(w, work=stand_ins.leave_an_index, fresh=True)   # --fresh: anew all the same
    _pump_until(lambda: not w.inprogress)
    assert "asked for (--fresh)" in capsys.readouterr().out
    assert jit_cache.why_compile(str(folder), cw.current_stamp()) is None
    monkeypatch.setattr(cw, "app_version", lambda: "99.0")               # a new version: anew
    assert cw.compile_if_needed(w, work=stand_ins.fail)
    _pump_until(lambda: not w.inprogress)
    assert w.log.toPlainText().startswith("Compiling the functions failed: RuntimeError: no compiler today")
    assert "Traceback" in capsys.readouterr().out                        # the whole of it, for the bug report
    assert jit_cache.why_compile(str(folder), cw.current_stamp()) == "nothing compiled yet"   # tried again next


def test_the_fit_the_two_starts_time_succeeds(qapp, tmp_path):
    """syncmoss_test.timed_fit, the plain fit main.py's --test times before
    the compiling and --test-again at a second start: here with a ThreadPool,
    on a temp copy of the parameters. It raises when the fit fails."""
    pool = ThreadPool(2)
    try:
        assert syncmoss_test.timed_fit(pool, tmp_path) > 0
    finally:
        pool.terminate()
        pool.join()


def test_closing_the_program_stops_the_compiling(qapp):
    FIT_CANCEL.clear()
    thread = cw.CompileThread(stand_ins.sleep_long)
    got = []
    thread.done.connect(got.append)
    thread.start()
    time.sleep(0.5)
    started = time.monotonic()
    thread.stop()                                   # what the program's end calls
    assert not thread.isRunning() and time.monotonic() - started < 30
    qapp.processEvents()
    assert got == ["Interrupted by the user"]
