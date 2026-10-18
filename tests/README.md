# SYNCmoss tests

Run from the `syncmoss/` folder with the project's environment:

```
.venv/Scripts/python.exe -m pytest                     # the whole suite, in parallel
.venv/Scripts/python.exe -m pytest tests/test_a.py     # some files
.venv/Scripts/python.exe -m pytest -m "not slow"       # what CI runs on every push
.venv/Scripts/python.exe -m pytest -n 0 -s tests/test_a.py::test_b   # debugging: serial, live output
```

## Parallel runs

`pyproject.toml` asks for `-n auto --dist loadgroup --no-loadscope-reorder`.

**How many workers** (`worker_plan.py`): the smallest of

* **memory** — what is available *now*, minus what stays free for the rest of
  the machine (the larger of 1.5 GB and 5 % of RAM; `SYNCMOSS_TEST_KEEP_FREE_GB`
  sets it), divided by the memory of a worker (`worker_mb`), after setting aside
  what the most demanding tests need on top of a typical one, as if they ran at
  the same moment;
* **processors** — at most one worker per logical processor, leaving two to the
  machine. Other programs' load does not lower the number (more workers than
  free processors only slows a run; too little memory breaks other work). Other
  Python processes that are computing — another chat's long run — are named in
  a WARNING line: wait for them if they matter;
* **work** — at least 10 s of tests per worker (a worker needs a few seconds to
  start).

The first line of the output says what was chosen and why; the last line the
peak memory the run really took. `-n N` (or `PYTEST_XDIST_AUTO_NUM_WORKERS=N`)
still forces a number. Without `psutil` the planner cannot see free memory and
uses 6.

**Which test goes where**: every worker sorts the tests longest first
(`order_for_workers`), and xdist hands them out in that order as workers become
free. The long tests start at once and the short ones fill in at the end.
Tests sharing an expensive module fixture carry the same `xdist_group` mark
(see `test_calibration.py`): they go to one worker together, so the fixture
runs once.

Each worker runs OpenBLAS with one thread: it does one test at a time, and a
thread per processor in every worker only oversubscribes the machine.

## The time and memory profile

`resource_profile.json` holds, per test, the seconds it ran and the memory it
held only while running. `worker_plan.py` reads it. Refresh it after adding or
changing tests — tests it does not know count as the median of their file:

```
.venv/Scripts/python.exe tests/profile_resources.py              # all files, ~10 min
.venv/Scripts/python.exe tests/profile_resources.py test_a.py    # some files
```

Each file runs alone in a fresh process; memory is sampled from outside the
process (numba code can hold the GIL for seconds).

## Writing a test

* **Look first.** Grep `tests/` for the function and the scenario. If a test with
  the same setup exists, add the assertion there; do not write a second one that
  checks a subset of it.
* **The lightest fixture that works.** A pure function needs no window. A window
  costs ~2 s to build and destroy, so:
  * `physics_app` — a fresh window, for a test that presses a button, logs,
    changes the table or fits;
  * `shared_physics_app` — ONE window per module, for many cases that only READ
    it (a parser fed through the path box, the menus, shipped documents). Mark
    them `pytest.mark.xdist_group("<name>")` too, or a parallel run hands them
    to many workers and each builds its own window. Keep them parametrized, so a
    failure still names its case.
* **Seconds, not minutes.** Run the real code path with a shorter SCHEDULE
  (monkeypatch the iteration/pass constants, `JN` 16, one spectrum, no
  escalation) — never fake the physics. A module's validation suite runs in its
  quick mode (`sms_theory.selftest(quick=True)`: the same checks on the coarsest
  grids that give the same answers). Mark `slow` only what cannot be shortened
  (real calibrations, the GUI smoke flow).
* **An expensive module fixture** (a calibration, a search) shared by several
  tests: give those tests one `xdist_group`, so it runs once per run.
* **Pools:** `SerialPool` (conftest) for the transmission integral; a
  `ThreadPool` when a real thread matters; a real `mp.Pool(1)` only for a test
  about processes.
* **Data:** only what is in the repository; numbers that depend on the
  calibration or the instrumental function come from `tests/data/parameters/`
  (`FROZEN_PARAMETERS`), never from `syncmoss/parameters/`. Call
  `redirect_params_dir_to_tmp` / `redirect_calibration_to_tmp` before anything
  that searches, calibrates or saves. Editable presets (`Be.txt`, `KB.txt`) are
  READ with the code's own reader, never typed into the test.
* **It must be able to fail.** Compare with an independent reference (a
  hand-written expansion, a closed form, a golden file) and assert the case is
  not vacuous (the distribution really changed the spectrum). Cover both
  directions of a pair (Find and Refine, forward and reversed spectra).
* **Leave nothing behind.** Every extra window a test opens is destroyed in the
  test; a Qt object the APPLICATION creates must have a parent (a parentless
  validator per field leaked ~10 MB per window until 2026-10).
* **Unique file names** across `tests/` and `tests_time/` (no `__init__.py`).
* **Afterwards:** refresh the profile for the files you touched
  (`tests/profile_resources.py test_x.py`); until then a new test counts as the
  median of its file.

## Rules the suite keeps

* Temporary files go under `.pytest_tmp/pytest-of-<user>/pytest-<N>` — numbered
  folders, so two runs at the same time (two chats) do not delete each other's.
* No test writes into `syncmoss/parameters/` (a session fixture checks it), and
  no test depends on what is in it: tests compute with the frozen copies in
  `tests/data/parameters/` (`redirect_params_dir_to_tmp`,
  `redirect_calibration_to_tmp`).
* No test reads data from outside the repository.
* `SerialPool` (conftest) stands in for the multiprocessing pool.
* `slow` marks the real calibrations, the full GUI smoke flow and a few real
  fits: CI's push lane skips them, the release lane runs everything.
