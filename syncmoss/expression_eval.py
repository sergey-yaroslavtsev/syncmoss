"""The user-typed texts: Expression, Distr (probability density) and Corr
(dependency) rows, and the one function that evaluates them.

The ``from numpy import (...)`` block below IS the set of bare names those
texts may use, e.g. "sqrt(p[3])" or "exp(-X**2/2)"; any other numpy function
is written ``np.<name>``. Every place that evaluates such a text calls
``eval_expr`` -- the fit, Show model, the check before them, the distribution
plot, the results table -- so they all accept the same texts. Nothing else
belongs in this module: its namespace is what the texts see.
"""
import numpy as np
from numpy import (
    # constants
    pi, e,

    # math basics
    exp, log, log10, sqrt, abs, power,

    # trig functions (core)
    sin, cos, tan,
    arcsin, arccos, arctan,
    sinh, cosh, tanh,
    arcsinh, arccosh, arctanh,

    # utility math
    floor, ceil, round, sign,

    # aggregation
    mean, std, var,
)


def eval_expr(expr, p, X=None, O=None):
    """Evaluate a user-typed text.

    ``p`` is the current parameter array, so the text may reference ``p[i]``;
    bare numpy functions (``sqrt``, ``exp``, ...) resolve through this module's
    globals (the imports above). ``X`` is optional: the distribution axis,
    given for the Distr/Corr texts. ``O`` is optional too: models.TImod's outer
    parameter array, which the Distr/Corr texts of a nested (multidimensional)
    call read as ``O[i]`` (TImod rewrites their ``p[`` to ``O[``)."""
    locals_dict = {"p": p}
    if X is not None:
        locals_dict["X"] = X
    if O is not None:
        locals_dict["O"] = O
    return eval(expr, globals(), locals_dict)
