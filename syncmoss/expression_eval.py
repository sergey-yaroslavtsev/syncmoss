"""The user-typed texts: Expression, Distr (probability density) and Corr
(dependency) rows, and the one function that evaluates them.

The ``from numpy import (...)`` block below and ``erf`` ARE the set of bare
names those texts may use, e.g. "sqrt(p[3])" or "exp(-X**2/2)"; any other
numpy function is written ``np.<name>``. Every place that evaluates such a text calls
``eval_expr`` -- the fit, Show model, the check before them, the distribution
plot, the results table -- so they all accept the same texts. Nothing else
belongs in this module: its namespace is what the texts see.

Every function below takes dual numbers (dual_numbers' tables), so a fit's
exact Jacobian and the results table's errors cover it; a function added here
gets its derivative there. No name may be one a text reads otherwise (p, X,
N, N1, ...) nor hide a Python builtin other than abs and round: sum, max, min
keep Python's meaning (maximum, minimum are the element-wise ones).
"""
import numpy as np
from numpy import (
    # constants
    pi, e,

    # math basics
    exp, log, log10, sqrt, abs, power,
    log2, expm1, log1p, cbrt, square, hypot,

    # trig functions (core)
    sin, cos, tan,
    arcsin, arccos, arctan, arctan2,
    sinh, cosh, tanh,
    arcsinh, arccosh, arctanh,
    deg2rad, rad2deg,

    # utility math
    floor, ceil, round, sign,
    maximum, minimum, where,

    # aggregation
    mean, std, var,
)

from syncmoss.dual_numbers import chain as _chain, value as _value
from syncmoss.faddeeva import faddeeva as _faddeeva


def erf(x):
    """The error function, from the Faddeeva function of the line shapes:
    erf(x) = 1 - exp(-x^2) w(i|x|), odd. Within 1e-15 of the exact one
    (absolute: below |x| ~ 1e-3, where erf is that small, not to all its
    digits). Takes dual numbers: its derivative is 2 exp(-x^2) / sqrt(pi)."""
    v = np.asarray(_value(x), dtype=float)
    a = np.abs(v)
    y = np.sign(v) * (1.0 - np.exp(-a * a) * _faddeeva(1j * a).real)
    return _chain(y, (2.0 / np.sqrt(pi) * np.exp(-v * v),), (x,))


def eval_expr(expr, p, X=None):
    """Evaluate a user-typed text.

    ``p`` is the model's parameter array, so the text may reference ``p[i]``;
    bare numpy functions (``sqrt``, ``exp``, ...) resolve through this module's
    globals (the imports above). ``X`` is optional: the distribution axis,
    given for the Distr/Corr texts."""
    locals_dict = {"p": p}
    if X is not None:
        locals_dict["X"] = X
    return eval(expr, globals(), locals_dict)
