"""What a start compiles when the cache folder is not complete for the running
program (compile_window, jit_cache): numba compiles the program's calculation
functions then and keeps them, instead of during the first calculation of each
model (up to a quarter of a minute each, about a minute for all).

numba compiles a function for the TYPES of its arguments, so everything here
goes through the program's own entry points with the arguments they get in use:
models.TI, the spectrum, SMS and CMS; models_dual.fit_jacobian, its exact
derivatives as a fit takes them; models_positions.mod_pos, the lines on the
plot; the components search's warm-up, its kernels and minimi_hi's linear
solve. Every fittable model once. Checked 2026-11-14: after these, the channel
average, the CMS cosine smearing, Distr / Corr / Recon / Distr_T / Layer rows, a
Spectrum row and Nbaseline sections of both methods compile nothing new
(tests/test_jit_cache.py). 96 points and 16 nodes: numba's code is the same
for any size.

It runs in a new process (jit_cache.run_in_new_process): numba neither compiles
nor stores again a function that its process already holds.
"""
import numpy as np

import syncmoss.components_search as components_search
import syncmoss.models as m5
import syncmoss.models_dual as md
from syncmoss.Calibration import MulCoCMS
from syncmoss.constants import SMS_POL_DEFAULT
from syncmoss.models_positions import mod_pos

X = np.linspace(-10.0, 10.0, 96)          # mm/s
JN = 16
#           Ns     Os   c2    lin    Nnr   Os'  c2'   lin'
BASELINE = [1.0e4, 0.3, 0.02, -0.15, 3.0e3, -0.2, 0.01, 0.1]
# One row of every fittable model: inside its bounds and away from the values
# its code takes a shortcut at (no field, a powder, an axis along the beam)
ROWS = {
    'Singlet': [3.0, 0.4, 0.098, 0.15],
    'Doublet': [6.0, 0.2, 0.8, 0.098, 0.2, 35.0, 50.0, 0.6, 1.1],
    'Sextet': [8.0, 0.1, -0.05, 33.0, 0.098, 0.12, 40.0, 60.0, 0.6, 0.5, 0.02, -0.01, 0.3, 3.0],
    'MDGD': [8.0, 0.1, -0.05, 33.0, 0.098, 0.12, 0.3, 0.1, 0.1, 0.1, 40.0, 60.0, 0.6, 0.5, 0.02, -0.01, 3.0],
    'Relax_MS': [5.0, 0.1, -0.05, 33.0, 0.098, 40.0, 60.0, 0.6, 0.5, 1.0, 5.0],
    'Relax_2S': [5.0, 0.1, -0.05, 33.0, 0.05, 0.05, -30.0, 0.12, 40.0, 60.0, 0.6, 0.5, 0.3, 1.2],
    'Hamiltonian': [5.0, 0.1, 0.4, 33.0, 0.098, 0.12, 0.3, 20.0, 30.0, 40.0, 50.0, 70.0, 0.65, 0.4, 0.55],
    'ASM': [5.0, 0.1, 0.02, 0.2, 30.0, 5.0, 0.098, 0.12, 0.3, 35.0, 50.0, 0.6, 25.0, 3.0, 20.0],
    'SCDW': [5.0, 0.1, 0.05, 30.0, 0.3, 0.2, 35.0, 50.0, 0.7, 0.6, 0.003, 0.005, 25.0, 5.0, 1.5, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 12.0, 3.0],
}
# The slot of a row that is a count (its fix box is locked): never fitted
COUNTS = {'Relax_MS': 10, 'ASM': 12, 'SCDW': 25}
# The two sources: SMS an instrumental function of three Gaussians (as
# INSexp.txt) with its MulCo and x0 (INSint.txt); CMS the source's G
SOURCES = (
    dict(Met=0, INS=np.array([0.111, -0.117, 0.467, 0.199, 0.068, 0.787, 0.514, -0.079, 0.404]), MulCo=2.65,
         x0=0.005, Norm=1.0),
    dict(Met=1, INS=0.097, MulCo=MulCoCMS, x0=0.0, Norm=0.995),
)
NO_LINKS = np.array([[-1], [-1], [-1]])   # fitting_io's confu without links


class _SerialPool:
    """TI's pool, run in this process: what it compiles is what is stored."""

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


def compile_functions(report=None):
    """Run every calculation function once (see the module's notes). Before
    each step *report* gets (steps done, steps, what is compiled now)."""
    pool = _SerialPool()
    steps = len(ROWS) + 1
    for done, (name, row) in enumerate(ROWS.items()):
        if report is not None:
            report(done, steps, name)
        model = [name]
        p = np.array(BASELINE + row, dtype=float)
        fix = [len(BASELINE) + COUNTS[name]] if name in COUNTS else []
        for s in SOURCES:
            m5.TI(X, p, model, JN, pool, s['x0'], s['MulCo'], s['INS'], Met=s['Met'], Norm=s['Norm'],
                  pol=SMS_POL_DEFAULT)
            prepared = md.prepare(X, model, JN, s['x0'], s['MulCo'], s['INS'], s['Met'], s['Norm'], SMS_POL_DEFAULT)
            md.fit_jacobian(prepared, pool, [0], [0], [0], fix, NO_LINKS, [], [])(p)
            mod_pos(p, model, s['INS'], s['Met'])
    if report is not None:
        report(len(ROWS), steps, "components search")
    components_search._warm_up()
