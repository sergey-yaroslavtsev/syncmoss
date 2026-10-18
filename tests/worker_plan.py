"""How many xdist workers a test run gets, and the order the tests go out in.

pyproject.toml runs pytest with ``-n auto --dist loadgroup``. For ``auto``,
xdist asks conftest.py's ``pytest_xdist_auto_num_workers``, which answers with
plan_workers(); every worker then sorts its tests with order_for_workers().
Both read resource_profile.json, written by profile_resources.py: for every
test the seconds it ran and the memory it held only while it ran (its window,
its arrays), and ``worker_mb``, the memory of a worker running a typical test.

The number of workers is the smallest of three limits:

* memory -- every worker needs ``worker_mb``. On top comes the larger of two
  things that do not happen together: the n tests that need the most beyond a
  typical one (``typical_test_mb``), assumed to run at the same moment -- the
  longest go first, and the heavy ones are long, so that is at the start; and
  ``accumulated_mb``, what all the workers gather over a whole run (in
  proportion to the selected work), which is at the end. Measured against real
  runs this overestimates by 15-25 %. Memory available now, minus what stays
  free for the rest of the machine (keep_free), has to cover that;
* processors -- at most one worker per logical processor, leaving SPARE_CPUS
  to the machine. What other programs use right now does NOT lower it: the
  operating system's own background load is small and spread thin, and more
  workers than free processors only slows a run down, while running out of
  memory breaks other people's work. Other PYTHON processes that are computing
  (another chat's long fit, a study) are named in a warning instead, so the
  person or chat starting the tests can wait for them to end;
* work -- a worker needs a few seconds to start, so each one should get at
  least MIN_WORK_S seconds of tests.

One worker is no gain over none, so that case runs serially.
"""
import json
import os
import statistics
import time
from pathlib import Path

PROFILE = Path(__file__).with_name("resource_profile.json")
MB = 2 ** 20
GB = 2 ** 30

# Memory left to the rest of the machine: the larger of these two. The
# environment variable SYNCMOSS_TEST_KEEP_FREE_GB replaces both.
KEEP_FREE_MIN = 1.5 * GB
KEEP_FREE_FRACTION = 0.05
# Processors left to the rest of the machine.
SPARE_CPUS = 2
# Another Python process using at least this many processors is "computing".
BUSY_PYTHON_CORES = 0.5
# Seconds of tests a worker should get to pay for its start-up.
MIN_WORK_S = 10.0
# The controller process (imports conftest, not the application).
CONTROLLER_MB = 100
# Used when psutil is missing: free memory cannot be seen then.
FALLBACK_WORKERS = 6


def load_profile():
    """resource_profile.json as a dict (empty if missing or unreadable)."""
    try:
        with open(PROFILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _medians(tests):
    """Median (seconds, mb) over *tests*, an iterable of [seconds, mb]."""
    tests = list(tests)
    if not tests:
        return 1.0, 50.0
    return (statistics.median(t[0] for t in tests),
            statistics.median(t[1] for t in tests))


def _plain_nodeid(nodeid):
    """*nodeid* without the ``@group`` suffix xdist adds under loadgroup."""
    at = nodeid.rfind("@")
    return nodeid[:at] if at > nodeid.rfind("]") else nodeid


def _file_of(nodeid):
    return nodeid.split("::", 1)[0]


def estimate(nodeid, tests, by_file):
    """[seconds, mb] of a test: measured, else the median of its file's tests,
    else the median of all tests (a new file)."""
    nodeid = _plain_nodeid(nodeid)
    if nodeid in tests:
        return tests[nodeid]
    return by_file.get(_file_of(nodeid)) or by_file[None]


def file_medians(tests):
    """{file: (median seconds, median mb)}, and {None: suite-wide medians}."""
    files = {}
    for nodeid, t in tests.items():
        files.setdefault(_file_of(nodeid), []).append(t)
    by_file = {f: _medians(ts) for f, ts in files.items()}
    by_file[None] = _medians(tests.values())
    return by_file


def _selection(config, tests, by_file):
    """[seconds, mb] of every profiled test the command-line paths select.

    -k and -m are applied later, at collection, so this is an upper bound. A
    path the profile does not know (a new test file) counts as ten tests of
    the suite's median size.
    """
    root = Path(str(config.rootpath)).resolve()
    here = Path(str(config.invocation_params.dir))
    chosen = {}
    for arg in config.args:
        path, _, rest = arg.partition("::")
        p = Path(path)
        if not p.is_absolute():
            p = here / p if (here / p).exists() else root / p
        p = p.resolve()
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if p.is_dir():
            prefix = "" if rel == "." else rel + "/"
            hits = [n for n in tests if n.startswith(prefix)]
        else:
            node = rel + ("::" + rest if rest else "")
            hits = [n for n in tests
                    if n == node or n.startswith(node + "::") or n.startswith(node + "[")]
            if not hits:
                chosen.update({f"{node}#{i}": by_file[None] for i in range(10)})
        chosen.update({n: tests[n] for n in hits})
    return list(chosen.values())


def busy_python_processes(psutil):
    """[(pid, processors in use)] of the OTHER Python processes computing now
    (sampled over 0.3 s; this run and its parents are left out)."""
    me = psutil.Process()
    own = {me.pid} | {p.pid for p in me.parents()}
    watched = []
    for p in psutil.process_iter(["name"]):
        if p.pid in own or not (p.info["name"] or "").lower().startswith("python"):
            continue
        try:
            p.cpu_percent(None)
            watched.append(p)
        except psutil.Error:
            pass
    time.sleep(0.3)
    busy = []
    for p in watched:
        try:
            cores = p.cpu_percent(None) / 100
        except psutil.Error:
            continue
        if cores >= BUSY_PYTHON_CORES:
            busy.append((p.pid, cores))
    return busy


def plan_workers(config):
    """(number of workers for ``-n auto``, what to print about it)."""
    try:
        import psutil
    except ImportError:
        n = min(FALLBACK_WORKERS, os.cpu_count() or 1)
        return n, f"workers: {n} (fixed number: psutil is not installed, free memory unknown)"

    profile = load_profile()
    tests = profile.get("tests", {})
    by_file = file_medians(tests)
    worker_mb = profile.get("worker_mb", 500)
    typical = profile.get("typical_test_mb", 150)
    selected = _selection(config, tests, by_file) if tests else []
    if not selected:            # no profile, or paths it cannot place
        selected = [by_file[None]] * 200

    memory = psutil.virtual_memory()
    keep_free = os.environ.get("SYNCMOSS_TEST_KEEP_FREE_GB")
    keep_free = (float(keep_free) * GB if keep_free
                 else max(KEEP_FREE_MIN, KEEP_FREE_FRACTION * memory.total))
    work = sum(t[0] for t in selected)
    whole = sum(t[0] for t in tests.values()) or work
    accumulated = profile.get("accumulated_mb", 0) * min(1.0, work / whole)
    budget = (memory.available - keep_free) / MB - CONTROLLER_MB
    extras = sorted((max(0.0, t[1] - typical) for t in selected), reverse=True)
    n_mem = 0
    while (n_mem < len(extras) and
           (n_mem + 1) * worker_mb + max(sum(extras[:n_mem + 1]), accumulated) <= budget):
        n_mem += 1

    n_cpu = max(1, (psutil.cpu_count() or 1) - SPARE_CPUS)
    n_work = max(1, int(work // MIN_WORK_S))

    n = min(n_mem, n_cpu, n_work, len(selected))
    n = n if n >= 2 else 0
    why = (f"workers: {n or 'none (serial)'}  -- memory: {memory.available / GB:.1f} GB available, "
           f"{keep_free / GB:.1f} GB kept free, ~{worker_mb / 1024:.2f} GB per worker -> room for "
           f"{n_mem}; processors: {n_cpu}; work: ~{work:.0f} s in {len(selected)} tests")
    busy = busy_python_processes(psutil)
    if busy:
        why += ("\nWARNING: other Python processes are computing ("
                + ", ".join(f"pid {pid}: {cores:.1f} processors" for pid, cores in busy)
                + "). The tests share the machine with them and slow them down; if that "
                  "job matters, wait until it ends.")
    return n, why


def order_for_workers(items):
    """Sort *items* longest first, in place.

    xdist (loadgroup, --no-loadscope-reorder) hands the tests out in this
    order as workers become free, so the long ones start at once and the
    short ones fill the gaps at the end. Tests sharing an ``xdist_group`` mark
    stay together and go out as one unit, timed by their sum.
    """
    tests = load_profile().get("tests", {})
    by_file = file_medians(tests)
    units = {}
    for item in items:
        mark = item.get_closest_marker("xdist_group")
        if mark:
            key = mark.args[0] if mark.args else mark.kwargs.get("name", "default")
        else:
            key = item.nodeid
        units.setdefault(key, []).append(item)
    seconds = {key: sum(estimate(it.nodeid, tests, by_file)[0] for it in unit)
               for key, unit in units.items()}
    order = sorted(units, key=lambda key: -seconds[key])   # stable: same order on every worker
    items[:] = [item for key in order for item in units[key]]
