"""Measure the time and memory of every test; write tests/resource_profile.json.

worker_plan.py reads that file to choose how many xdist workers fit in the
memory that is free, and to hand out the longest tests first. Refresh it after
adding or changing tests (it needs psutil; about 15 minutes for the suite, the
last 6 of them a parallel calibration run):

    cd syncmoss
    .venv/Scripts/python.exe tests/profile_resources.py                 # all files
    .venv/Scripts/python.exe tests/profile_resources.py test_a.py ...   # some files

Every test file runs alone in a fresh ``pytest -n 0`` process, so its numbers
do not depend on what ran before it. The resident memory of that process and
its children is sampled from OUTSIDE every 50 ms: numba code can hold the GIL
for seconds, and a sampling thread inside the process would miss exactly the
peaks that matter. Several files run side by side (--jobs), each with one
OpenBLAS thread, as in a parallel test run; a file is only started while at
least 2 GB are available.

Written per test: seconds, and the memory the test held only while it ran --
its peak above what it left behind (a GUI test's window, a test's arrays).
What a test leaves behind is mostly warm-up (Qt, matplotlib, the numba models
it used), paid once per worker, not per test. Written for the suite:
typical_test_mb, the 90th percentile of that per-test memory; worker_mb, the
memory of a worker running a typical test -- every test module imported (a
collection of the whole suite), the warm-up a file's tests leave behind (75th
percentile over files) and typical_test_mb; and accumulated_mb, what the
workers of a whole parallel run gather on top of that, together (caches, the
little each window leaves). Only a real run shows the last one -- a fresh
process per file overstates it many times, since most of what a file's tests
keep is warm-up again -- so the suite is finally run with CALIBRATION_WORKERS
workers and each worker's steady level is recorded (--no-calibration skips it).
"""
import argparse
import bisect
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROFILE = os.path.join(HERE, "resource_profile.json")
MB = 2 ** 20
MIN_AVAILABLE = 2 * 2 ** 30
TIMEOUT_S = 20 * 60
CALIBRATION_WORKERS = 4

# Loaded into each child pytest: one line per test boundary.
_EVENTS_PLUGIN = """
import json, os, time
_f = open(os.environ["PROFILE_EVENTS"], "a", buffering=1, encoding="utf-8")
def pytest_runtest_logstart(nodeid, location):
    _f.write(json.dumps(["start", nodeid, time.time()]) + "\\n")
def pytest_runtest_logfinish(nodeid, location):
    _f.write(json.dumps(["end", nodeid, time.time()]) + "\\n")
"""


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Run:
    """One pytest process with its memory samples (time, resident MB of the
    whole process tree) and, per process, its own samples."""

    def __init__(self, args, work, name, workers=0):
        self.name = name
        self.events = os.path.join(work, name + ".events")
        env = dict(os.environ, PROFILE_EVENTS=self.events, OPENBLAS_NUM_THREADS="1",
                   PYTHONPATH=work + os.pathsep + os.environ.get("PYTHONPATH", ""))
        env.pop("PYTEST_XDIST_AUTO_NUM_WORKERS", None)
        self.out = open(os.path.join(work, name + ".out"), "w", encoding="utf-8")
        self.t0 = time.time()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "pytest", *args, "-n", str(workers), "-q", "-p", "profile_events",
             "-p", "no:cacheprovider", "--basetemp", os.path.join(work, "tmp", name)],
            cwd=ROOT, env=env, stdout=self.out, stderr=subprocess.STDOUT)
        self.ps = psutil.Process(self.proc.pid)
        self.children, self.children_at = [], 0.0
        self.samples = []
        self.per_process = {}

    def sample(self, now):
        if now - self.children_at > 0.5:
            try:
                self.children = self.ps.children(recursive=True)
            except psutil.Error:
                self.children = []
            self.children_at = now
        total = 0
        for p in [self.ps] + self.children:
            try:
                rss = p.memory_info().rss
            except psutil.Error:
                continue
            total += rss
            self.per_process.setdefault(p.pid, []).append((now, rss / MB))
        if total:
            self.samples.append((now, total / MB))

    def steady_levels(self, workers):
        """Median resident MB over the second half of the life of each of the
        *workers* biggest processes (the xdist workers: the controller and the
        venv launchers are far smaller)."""
        levels = []
        for samples in self.per_process.values():
            life = samples[-1][0] - samples[0][0]
            later = [s[1] for s in samples if s[0] > samples[0][0] + life / 2]
            if later:
                levels.append(statistics.median(later))
        return sorted(levels, reverse=True)[:workers]

    def poll(self, now):
        rc = self.proc.poll()
        if rc is None and now - self.t0 > TIMEOUT_S:
            for p in self.children + [self.ps]:
                try:
                    p.kill()
                except psutil.Error:
                    pass
            rc = "timeout"
        if rc is not None:
            self.out.close()
        return rc


def run_all(jobs_args, jobs, work, workers=0):
    """Run every (name, args) with at most *jobs* side by side; {name: Run}."""
    pending, running, done = list(jobs_args), [], {}
    while pending or running:
        now = time.time()
        if pending and len(running) < jobs and psutil.virtual_memory().available > MIN_AVAILABLE:
            name, args = pending.pop(0)
            running.append(Run(args, work, name, workers))
        for run in list(running):
            run.sample(now)
            rc = run.poll(now)
            if rc is not None:
                running.remove(run)
                done[run.name] = run
                peak = max((s[1] for s in run.samples), default=0)
                log(f"{run.name}: rc={rc} {now - run.t0:.0f} s, peak {peak:.0f} MB")
        time.sleep(0.05)
    return done


def per_test(run):
    """{nodeid: (seconds, peak above the end, resident at start, at end)} of a run."""
    times = [s[0] for s in run.samples]
    rss = [s[1] for s in run.samples]

    def at(t):
        return rss[max(bisect.bisect_right(times, t) - 1, 0)]

    def peak(t0, t1):
        seg = rss[max(bisect.bisect_left(times, t0) - 1, 0):bisect.bisect_right(times, t1) + 1]
        return max(seg) if seg else 0.0

    out, starts = {}, {}
    if not rss or not os.path.exists(run.events):
        return out
    with open(run.events, encoding="utf-8") as f:
        for line in f:
            kind, nodeid, t = json.loads(line)
            if kind == "start":
                starts[nodeid] = t
            elif nodeid in starts:
                t0 = starts.pop(nodeid)
                out[nodeid] = (t - t0, peak(t0, t) - at(t), at(t0), at(t))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", help="test files to (re)measure; default: all")
    parser.add_argument("--jobs", type=int, default=4, help="files measured side by side")
    parser.add_argument("--no-calibration", action="store_true",
                        help="skip the parallel run that measures accumulated_mb (keep the old value)")
    opts = parser.parse_args()

    files = opts.files or sorted(f for f in os.listdir(HERE) if f.endswith(".py") and
                                 (f.startswith("test_") or f.endswith("_test.py")))
    files = [os.path.basename(f) for f in files]
    files.sort(key=lambda f: -os.path.getsize(os.path.join(HERE, f)))   # big ones first
    work = tempfile.mkdtemp(prefix="profile_resources_")
    os.makedirs(os.path.join(work, "tmp"))
    with open(os.path.join(work, "profile_events.py"), "w", encoding="utf-8") as f:
        f.write(_EVENTS_PLUGIN)

    log(f"{len(files)} files, {opts.jobs} side by side; scratch: {work}")
    jobs_args = [("collect", ["tests", "--collect-only"])]
    jobs_args += [(f[:-3], ["tests/" + f]) for f in files]
    runs = run_all(jobs_args, opts.jobs, work)

    profile = {}
    if os.path.exists(PROFILE):
        with open(PROFILE, encoding="utf-8") as f:
            profile = json.load(f)
    tests = {n: t for n, t in profile.get("tests", {}).items()
             if os.path.exists(os.path.join(ROOT, n.split("::")[0]))
             and os.path.basename(n.split("::")[0]) not in files}
    warmups = []
    for f in files:
        measured = per_test(runs[f[:-3]])
        for nodeid, (seconds, mb, _, _) in measured.items():
            tests[nodeid] = [round(seconds, 2), max(0, round(mb))]
        if measured:
            first_start = min(m[2] for m in measured.values())
            last_end = runs[f[:-3]].samples[-1][1] if runs[f[:-3]].samples else first_start
            warmups.append(max(0.0, last_end - first_start))
    collect = max((s[1] for s in runs["collect"].samples), default=0)
    typical = round(statistics.quantiles([t[1] for t in tests.values()], n=10)[8])
    warmup = statistics.quantiles(warmups, n=4)[2] if len(warmups) > 1 else 0
    worker_mb = round(collect + warmup + typical)
    if opts.files and "worker_mb" in profile:
        worker_mb = profile["worker_mb"]     # a few files say nothing about the suite's warm-up

    accumulated = profile.get("accumulated_mb", 0)
    if not opts.files and not opts.no_calibration:
        log(f"calibration: the whole suite on {CALIBRATION_WORKERS} workers")
        cal = run_all([("calibration", ["tests"])], 1, work, CALIBRATION_WORKERS)["calibration"]
        levels = cal.steady_levels(CALIBRATION_WORKERS)
        accumulated = round(max(0.0, sum(levels) - len(levels) * worker_mb))
        log(f"workers settle at {', '.join(f'{x:.0f}' for x in levels)} MB "
            f"-> accumulated_mb={accumulated}")

    profile = {
        "about": "Written by tests/profile_resources.py, read by tests/worker_plan.py. "
                 "tests: nodeid -> [seconds, MB of resident memory held only while the test "
                 "ran]. typical_test_mb: 90th percentile of that. worker_mb: resident memory "
                 "of a worker running a typical test. accumulated_mb: what all the workers of "
                 "a parallel run of the whole suite gather on top, together.",
        "measured": f"{time.strftime('%Y-%m-%d')} on {platform.system()} {platform.machine()}, "
                    f"{psutil.cpu_count(logical=False)}/{psutil.cpu_count()} cores, "
                    f"Python {platform.python_version()}, {opts.jobs} files side by side",
        "worker_mb": worker_mb,
        "typical_test_mb": typical,
        "accumulated_mb": accumulated,
        "tests": dict(sorted(tests.items())),
    }
    with open(PROFILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=0)
        f.write("\n")
    total = sum(t[0] for t in tests.values())
    log(f"wrote {PROFILE}: {len(tests)} tests, {total:.0f} s in total, worker_mb={worker_mb}, "
        f"typical_test_mb={typical}, accumulated_mb={accumulated}")


if __name__ == "__main__":
    main()
