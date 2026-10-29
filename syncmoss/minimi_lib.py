# -*- coding: utf-8 -*-
"""
@author: YAROSLAVTSEV S

The MIT license follows:

Copyright (c) European Synchrotron Radiation Facility (ESRF)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

################################################################################
SYNCmoss Levenberg-Marquardt least-squares minimiser.

Public API:
    minimi_hi(model_func, x_exp, y_exp, p0, fix=, confu=, bounds=, Expr=, NExpr=,
              MI=, MI2=, nu0=, tau0=, eps=, fixCH=, n_reg=, batch_eval=, weights=,
              chi2_correction=)
        -> (params, errors, chi2_red, covariance, chi2_pearson)
Damped Gauss-Newton (Levenberg-Marquardt) fit of ``model_func(x_exp, params)`` to
``y_exp``, with support for fixed parameters (``fix``), linear parameter coupling
(``confu``), box ``bounds`` and linked expressions (``Expr`` evaluated into the
``NExpr`` indices).

Implementation notes:
  * The numba linear-algebra kernel ``solution`` is ``@njit(cache=True)``, so
    its one-time compilation is cached on disk and reused across launches /
    multiprocessing workers; ``warmup()`` lets the application trigger that
    compile at a predictable moment.
  * The gradient, the damped normal-equation assembly, the chi-square sums and the
    Jacobian assembly are vectorised.
  * The parameter covariance is that of the UNWEIGHTED fit (the sandwich, see
    step 5 of ``minimi_hi``), so no variance can come out negative; a parameter
    that the others reproduce is reported like a fixed one (``_determined_rows``).
  * A trial step whose model output is non-finite is rejected (damping increased)
    rather than crashing the fit.
  * Variable names are descriptive; the textbook LM symbols map as
    mu->damping, tau->damping_scale, nu->damping_growth, rho->gain_ratio,
    chi->residual vector (y - model), J->jac, A->hessian, g->grad.
"""

import numpy as np
from numba import njit

# The user's Expr strings are evaluated by expression_eval.eval_expr (its bare
# numpy names + p); our own code calls numpy explicitly as ``np.<func>``.
from syncmoss.expression_eval import eval_expr


# =============================================================================
# Linear-algebra kernels (numba, cache=True)
# =============================================================================
# ``cache=True``(so the compiled machine code is written to disk
# and reused across launches and multiprocessing workers
# no per-process recompilation).

@njit(cache=True)
def solution(aug):
    """Solve the linear system whose augmented matrix is ``aug = [M | b]`` (last
    column is the right-hand side) by Gaussian elimination + back-substitution.
    Returns the solution vector. ``aug`` is modified in place."""
    n = len(aug)
    x = np.array([float(0)] * n)
    for i in range(0, n):
        # find a pivot row at/below i with a non-zero entry in column i
        k = i
        for j in range(i, n):
            if np.abs(aug[j][i]) > 0:
                k = j
                break
        if i != k:                                  # swap rows i and k
            tmp = np.array([float(0)] * (n + 1))
            for m in range(0, n + 1):
                tmp[m] = aug[i][m]
                aug[i][m] = aug[k][m]
                aug[k][m] = tmp[m]
        for m in range(i + 1, n):                   # eliminate below the pivot
            factor = aug[m][i]
            if (np.abs(factor) > 0):
                for col in range(i, n + 1):
                    aug[m][col] = aug[m][col] - factor / aug[i][i] * aug[i][col]
    for i in range(0, n):                           # back-substitution
        x[n - 1 - i] = aug[n - 1 - i][n] / aug[n - 1 - i][n - 1 - i]
        for j in range(0, i):
            x[n - 1 - i] += -aug[n - 1 - i][n - 1 - j] * x[n - 1 - j] / aug[n - 1 - i][n - 1 - i]
    return (x)


def warmup():
    """Trigger (or load from cache) the JIT compilation of the kernel above.

    Call this ONCE at application start (e.g. while a splash screen is up) so the
    one-time numba compile happens at a predictable moment instead of inside the
    user's first fit. With ``cache=True`` the compiled code is reused on every
    later launch and in every pool worker, so this becomes a fast cache-load
    rather than a recompile. Safe to call multiple times.
    """
    # A dense, well-conditioned system, like a real LM step.
    aug = np.array([[2.0, 1.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    solution(np.copy(aug))


def _solve_lm(hessian, damping, grad):
    """Return the LM step solving ``(hessian + diag(damping)) . step = grad``.

    Builds the augmented matrix ``[hessian + diag(damping) | grad]`` with array
    assignments (instead of the original's Python double loop) and hands it to the
    numba ``solution`` kernel. ``hessian`` is left untouched (damping is added to a
    copy).
    """
    dim = len(hessian)
    aug = np.empty((dim, dim + 1), dtype=np.float64)
    aug[:, :dim] = hessian
    aug[:, dim] = grad
    diag = np.arange(dim)
    aug[diag, diag] += damping                      # add damping to the diagonal
    return solution(aug)


def norm(values):
    """Root-mean-square-like norm used throughout the optimiser.

    ``sqrt(sum((v / len(v))**2))`` — identical to the original ``norm`` (it divides
    each element by the length *before* squaring). Kept byte-for-byte so every
    convergence test compares exactly the same quantity as the original.
    """
    return np.sqrt(np.sum((np.array([values]) / len(values)) ** 2))


def poisson_deviance(y, m):
    """The Poisson deviance 2 sum [m - y + y ln(y/m)] (y ln y = 0 at y = 0):
    -2 ln of the likelihood ratio against a model through every point; inf
    where the model is not positive and finite (such a step is rejected).
    ln y - ln m rather than ln(y/m): y/m overflows for a tiny m."""
    if not np.all(np.isfinite(m) & (m > 0)):
        return np.inf
    counted = y > 0
    return 2 * np.sum(m - y + np.where(counted, y * (np.log(np.where(counted, y, 1.0)) - np.log(m)), 0.0))


def chi2_spread(n_points, n_free):
    """The 1-sigma spread a reduced chi-square (or deviance) has when the
    model is RIGHT.

    ``sqrt(2/dof)``. Without it a user cannot tell whether 1.3 is a bad fit or
    an ordinary fluctuation: on 40 degrees of freedom 1.3 is one sigma and
    means nothing, on 4000 it is nine sigma and means the model is wrong.
    """
    dof = max(int(n_points) - int(n_free), 1)
    return float(np.sqrt(2.0 / dof))


# =============================================================================
# Jacobian by forward finite differences
# =============================================================================
# The relative step of the finite differences is 1/JACOBIAN_STEP_DIVISOR; it
# is also the resolution of the Jacobian that _determined_rows works to.
JACOBIAN_STEP_DIVISOR = 10 ** 6


def _bump_parameter(params, k, perturbation, confu, Expr, NExpr):
    """``params`` with parameter *k* stepped by *perturbation*, coupling honoured.

    Split out of ``compute_jacobian``'s loop (as on the SYNCtime branch) so all
    the bumped vectors can be built before any of them is evaluated; the linked
    expressions and the linear constraints are propagated exactly as the
    original in-line code did.
    """
    bumped = np.copy(params)
    bumped[k] = bumped[k] + perturbation

    # Re-evaluate linked expressions after the bump and propagate their change
    # through any linear constraints attached to the expression target.
    for e_i in range(0, len(Expr)):
        bumped[NExpr[e_i]] = eval_expr(str(Expr[e_i]), bumped)
        expr_delta = bumped[NExpr[e_i]] - params[NExpr[e_i]]
        if np.any(confu[1] == NExpr[e_i]):
            matches = np.where(confu[1] == NExpr[e_i])[0]
            for c_i in range(0, len(matches)):
                bumped[int(confu[0][matches[c_i]])] = bumped[int(confu[0][matches[c_i]])] + expr_delta * confu[2][matches[c_i]]

    # If the bumped parameter is itself the source of a linear constraint (and
    # is not driven by an expression), propagate its bump to the constrained
    # parameters too.
    if np.any(confu[1] == k) and not np.any(NExpr == k):
        matches = np.where(confu[1] == k)[0]
        for c_i in range(0, len(matches)):
            bumped[int(confu[0][matches[c_i]])] = bumped[int(confu[0][matches[c_i]])] + perturbation * confu[2][matches[c_i]]
    return bumped


def compute_jacobian(model_func, params, x_exp, fix, confu, model_base,
                     Expr, NExpr, verbose=False, batch_eval=None):
    """Forward-difference Jacobian of the model ``model_func(x_exp, params)``.

    For every free parameter ``k`` (not in ``fix``) we bump ``params[k]`` by a
    small ``perturbation`` and form the column ``(model_up - model_base) / perturbation``,
    where ``model_base`` is the model at the un-bumped parameters. Each free
    parameter therefore costs exactly one model evaluation — this loop dominates
    the whole fit, so the model-evaluation *count* is kept identical to the
    original (the only change is that rows are collected once into ``np.array``
    instead of grown with ``np.append`` per column).

    Parameter coupling (``Expr``/``NExpr`` linked expressions and ``confu`` linear
    constraints) is honoured while bumping, exactly as in the original.

    ``batch_eval``, when given, is called ONCE with the list of bumped parameter
    vectors and must return the model output for each, in order -- it is how a
    caller whose model is expensive gets the columns computed together (the same
    hook as on the SYNCtime branch). The vectors are pure bookkeeping (no model
    evaluation happens while they are built), so the evaluation COUNT and ORDER
    are identical to the serial path.

    Returns ``(jac, inactive)``:
      * ``jac``  : array (n_effective_free, n_data), one row per parameter that
                   actually moves the model.
      * ``inactive`` : indices of parameters whose column came out all-zero (they do
                   not affect the model); the caller excludes them from steps.
    """
    n_data = len(x_exp)
    jac_rows = []        # Jacobian rows for parameters that move the model
    inactive = []            # indices of "inactive" parameters (all-zero column)

    plan = []            # (k, perturbation, bumped) -- built without evaluating

    for k in range(0, len(params)):
        if np.any(fix == k):
            continue

        # ``perturbation`` is the tiny increment h added to parameter k to estimate
        # its derivative numerically:  dModel/dp_k ~= (model(p+h) - model(p)) / h.
        # h is chosen RELATIVE to the parameter (1e-6 * |p_k|) so it scales with the
        # parameter's magnitude, with a 1e-12 absolute floor so a parameter sitting
        # at exactly 0 still gets a non-zero bump.
        #   Why 1e-6 and not the textbook forward-difference optimum sqrt(eps)~1e-8?
        #   The model here is a numerical transmission integral (JN samples), so it
        #   carries its own discretisation noise; a smaller h would be swamped by
        #   that noise. 1e-6 trades a little derivative bias for robustness.
        #   Limitation: the 1e-12 floor is crude for a near-zero parameter with a
        #   small sensitivity (the forward difference can lose precision and the
        #   column may read as all-zero -> the parameter is reported inactive). A
        #   per-parameter "typical scale" would be the principled fix but needs
        #   problem-specific input the API does not currently take.
        perturbation = np.maximum(10 ** -12, np.abs(params[k]) / JACOBIAN_STEP_DIVISOR)
        plan.append((k, perturbation,
                     _bump_parameter(params, k, perturbation, confu, Expr, NExpr)))

    # One model evaluation -> one Jacobian column (forward difference). This is
    # the loop that dominates the whole fit, so it is the one worth batching.
    if batch_eval is not None and len(plan) > 1:
        model_ups = list(batch_eval([bumped for _k, _h, bumped in plan]))
    else:
        model_ups = [model_func(x_exp, bumped) for _k, _h, bumped in plan]

    for (k, perturbation, _bumped), model_up in zip(plan, model_ups):
        jac_col = (model_up - model_base) / perturbation

        if np.all(jac_col == 0):
            # Parameter does not influence the model here -> record it so the
            # caller can skip it; it must not enter the (singular) normal eqs.
            inactive.append(k)
            if verbose:
                print('parameter No. ' + str(k) + ' does not affect the model')
        else:
            jac_rows.append(jac_col)

    if len(jac_rows) == 0:
        jac = np.empty((0, n_data), dtype=float)
        if verbose:
            print('what am I supposed to do if everything is fixed?')
    else:
        jac = np.array(jac_rows, dtype=float)

    return jac, np.array(inactive, dtype=int)


# Backwards-compatible alias (the original module exposed this helper as ``Jac``).
Jac = compute_jacobian


# =============================================================================
# Parameter covariance
# =============================================================================
def _jacobian_parameters(n_params, fix, inactive):
    """The parameter of each row of ``compute_jacobian``'s ``jac``, in order."""
    return np.array([k for k in range(n_params)
                     if not np.any(fix == k) and not np.any(inactive == k)], dtype=int)


def _determined_rows(normal):
    """The rows of the normal matrix ``normal`` (= J.J^T) whose parameter the
    data determine.

    Cholesky of its unit-diagonal form, row by row in index order: a pivot is
    the share of a parameter's effect that the rows kept before it do not
    reproduce (1 = independent, 0 = a combination of them). A share below the
    Jacobian's resolution -- its relative step 1/JACOBIAN_STEP_DIVISOR, squared,
    as the pivot is a squared quantity -- means the parameter only repeats the
    others (two parameters entering as a product, an offset added to the
    baseline). It is left out: its error is nan as for a fixed one, so Take
    result fixes it, and the earlier ones keep the combination free.
    """
    scale = np.sqrt(np.diag(normal))
    unit = normal / np.outer(scale, scale)
    factor = np.zeros_like(unit)
    rows = []
    for i in range(len(unit)):
        pivot = unit[i, i] - np.dot(factor[i, :i], factor[i, :i])
        if pivot > 1 / JACOBIAN_STEP_DIVISOR ** 2:
            factor[i, i] = np.sqrt(pivot)
            factor[i + 1:, i] = (unit[i + 1:, i] - np.matmul(factor[i + 1:, :i], factor[i, :i])) / factor[i, i]
            rows.append(i)
    return rows


# =============================================================================
# Levenberg-Marquardt least-squares fit  (public API — unchanged)
# =============================================================================
def minimi_hi(model_func, x_exp, y_exp, p0, fix=np.array([], dtype=int),
              confu=np.array([[-1], [-1], [0]], dtype=float),
              bounds=np.array([[], []], dtype=float), Expr=[],
              NExpr=np.array([], dtype=int), MI=10, MI2=20, nu0=2.618,
              tau0=0.001, eps=10 ** -10, fixCH=0, n_reg=0, batch_eval=None, weights='U',
              chi2_correction=True):
    """Levenberg-Marquardt least-squares fit of ``model_func(x_exp, params)`` to ``y_exp``.

    Optimised twin of the previous ``minimi_lib_old.minimi_hi`` — identical signature
    (keyword names unchanged: ``fix, confu, bounds, Expr, NExpr, MI, MI2, nu0,
    tau0, eps, fixCH``), identical algorithm and return value, faster internals
    and a non-finite-result guard (see the module docstring). Supports fixed
    parameters (``fix``), linear parameter coupling (``confu``), box ``bounds``
    and linked expressions (``Expr`` evaluated into the ``NExpr`` indices).
    ``MI`` / ``MI2`` cap the outer/inner iterations; ``eps`` is the tolerance.

    ``batch_eval`` (default None) offers the JACOBIAN's model evaluations up in
    one list instead of one at a time, so an expensive model can compute the
    columns together; see :func:`compute_jacobian`. It does not change the
    number of evaluations.

    ``n_reg`` (default 0, fully backward compatible) is the number of TRAILING
    rows of ``y_exp`` / the model output that are Tikhonov-style regularization
    pseudo-observations rather than real data (used by the ``Recon`` distribution
    reconstruction). Those rows participate normally in the LM step, gradient and
    Hessian (that is how the penalty pulls the fit), but are EXCLUDED from the
    reported reduced chi-square, the degrees of freedom and the data part of the
    parameter covariance (they carry no noise; in H they act as in the fit).

    ``weights`` is how the data rows weigh in the fit (docs/minimization.md);
    the variance it implies is the one of the chi-square and of the errors:
      * ``'U'`` (default; ``None`` too) -- unweighted, variance |y|+1;
      * ``'W'`` -- weights 1/(|y|+1), the chi-square it reports is the sum it
        minimises; biased when a point has fewer counts than there are points;
      * ``'P'`` -- the Poisson likelihood, minimised directly: every step is
        its Fisher-scoring step (weights 1/m of the model where the fit stands,
        from the start on) and is kept only if the deviance drops; variance m,
        the chi-square returned the reduced deviance; data and model must be
        counts (>= 0, model > 0);
      * one positive weight per data row -- fixed weights, variance 1/weight.
    Penalty rows (``n_reg``) keep the weight 1.

    ``chi2_correction`` (default True) multiplies the covariance -- so the
    errors -- by Pearson's reduced chi-square, the last value returned, for
    every ``weights``, below 1 as well as above; where that is nan (a model
    that is not positive: not counts) or not above ``eps`` (0 to the fit's
    tolerance: a spectrum without noise) nothing multiplies it. False: the
    errors the variances alone give (for 'P' the Cramer-Rao ones).

    Returns:
        tuple ``(params, errors, chi2_red, covariance, chi2_pearson)`` - fitted
        parameters, their 1-sigma errors (``nan`` for fixed, inactive and
        undetermined parameters), the reduced chi-square the fit minimised (for
        'P' the deviance per degree of freedom), the parameter covariance: one
        row per parameter whose error is not nan, in index order, scaled like
        the errors (errors = sqrt of its diagonal) -- and Pearson's reduced
        chi-square, sum (y-m)^2/m, ``nan`` where the model is not positive.
    """
    # ----------------------------------------------------------------------- #
    # 0) Set-up: coerce inputs, initialise the LM state machine                #
    # ----------------------------------------------------------------------- #
    x_exp = np.array(x_exp, dtype=float)
    y_exp = np.array(y_exp, dtype=float)
    fix_orig = np.copy(fix)                   # remember the user's original fixes
    # Number of REAL data rows (the trailing ``n_reg`` rows are regularization
    # pseudo-observations). Used for every reported/statistical quantity; the LM
    # step itself still uses all rows. n_reg == 0 reproduces the original path.
    n_data = len(y_exp) - int(n_reg)

    # A weighted fit is the unweighted fit of y*sqrt(w) by model*sqrt(w): every
    # step, the chi-square and the covariance below then work on scaled rows,
    # whose variance is 1. Unweighted, nothing is scaled and the variance of a
    # point is |y|+1 (the estimate of the reported chi-square). 'P' takes 1/m
    # of the model wherever the fit stands: at the start (step 1) and after
    # every accepted step (step 3).
    poisson = isinstance(weights, str) and weights == 'P'
    if poisson and np.any(y_exp[:n_data] < 0):
        raise ValueError("weights='P' needs counts: the data are negative somewhere")
    if weights is None or (isinstance(weights, str) and weights == 'U'):
        scale = None
    elif isinstance(weights, str) and weights == 'W':
        scale = 1 / np.sqrt(np.abs(y_exp[:n_data]) + 1)
    elif poisson:
        scale = np.ones(n_data)                 # set from the start model in step 1
    elif isinstance(weights, str):
        raise ValueError("weights is 'U', 'W', 'P', None or one weight per data point, not %r" % weights)
    else:
        given = np.asarray(weights, dtype=float)
        if given.shape != (n_data,) or not np.all(np.isfinite(given) & (given > 0)):
            raise ValueError('weights: one positive weight per data point (%d) expected' % n_data)
        scale = np.sqrt(given)
    y_raw = y_exp
    if scale is not None:
        scale = np.concatenate([scale, np.ones(len(y_exp) - n_data)])   # penalty rows: weight 1
        y_exp = y_exp * scale
    variance = np.abs(y_exp[:n_data]) + 1 if scale is None else np.ones(n_data)

    # Some SYNCmoss models (e.g. models.TI) return an ``object``-dtype array. The
    # numba kernels need real float64, so wrap the user model once to coerce every
    # output to float64 (lossless — the objects are Python floats), and to scale
    # it like y_exp. The wrapper forwards to the raw model so any caller-side call
    # counter still sees exactly one call per evaluation.
    _raw_model = model_func

    if scale is None:
        def model_func(_x, _p):
            return np.asarray(_raw_model(_x, _p), dtype=float)
    else:
        def model_func(_x, _p):
            return np.asarray(_raw_model(_x, _p), dtype=float) * scale

    _raw_batch = batch_eval
    if _raw_batch is not None:                 # the same coercion for a batch
        if scale is None:
            def batch_eval(_plist):
                return [np.asarray(m, dtype=float) for m in _raw_batch(_plist)]
        else:
            def batch_eval(_plist):
                return [np.asarray(m, dtype=float) * scale for m in _raw_batch(_plist)]

    # --- Levenberg-Marquardt symbol map (textbook Greek -> name used here) --- #
    #   mu   ->  damping         per-parameter damping added to the Hessian diagonal
    #   tau  ->  damping_scale   scalar setting mu relative to diag(Hessian)
    #   nu   ->  damping_growth  factor mu is multiplied by when a step is rejected
    #   rho  ->  gain_ratio      actual / predicted chi^2 drop of a trial step
    #   chi  ->  chi             residual vector y - model (its sum of squares = chi^2)
    #   J    ->  jac             Jacobian; A -> hessian (J.Jt); g -> grad (J.chi)
    # ------------------------------------------------------------------------- #

    # --- LM bookkeeping / state-machine variables --------------------------- #
    stop_code = 0          # 0 = running; nonzero -> which stopping condition fired
    damping_scale = tau0   # current damping scale (tau): mu = damping_scale*diag(H)
    accepted_any = 0       # latch: at least one step has been accepted
    damping_growth = nu0   # factor (nu) by which damping grows on a rejected step
    bound_hit = 0.5        # index of a parameter that just hit a bound (0.5 = none)
    gain_ratio = float(0)  # rho: actual reduction / predicted reduction
    strategy_count = 0     # counts mu-strategy switches (diagonal <-> max-diagonal)
    stall_count = 0        # "no-move" counter; detects when the fit cannot progress
    damping_branch = 0     # which mu-update branch to take after an accepted step
    bound_flags = np.array([int(0)] * len(p0))   # per-parameter bound-violation flags
    params = np.copy(p0)

    # Default bounds = unbounded; validate user-supplied bounds.
    if len(bounds[0]) == 0:
        bounds = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)], dtype=float)
    if len(bounds[0]) != len(p0) and len(bounds[1]) != len(p0):
        print('error: number of bounds do not match number of parameters')
    for i in range(0, len(params)):
        if params[i] < bounds[0][i] or params[i] > bounds[1][i]:
            print('initial parameters are out of bounds (bounds are included)')

    # ----------------------------------------------------------------------- #
    # 1) Initial model, Jacobian, Hessian approximation and gradient          #
    # ----------------------------------------------------------------------- #
    model_cur = model_func(x_exp, params)     # model at the starting point
    if poisson:
        # The Poisson weights from the start: 1/m at p0, so the first step is
        # already the likelihood's (the scale is still 1: model_cur is the model)
        if not np.all(np.isfinite(model_cur[:n_data]) & (model_cur[:n_data] > 0)):
            raise ValueError("weights='P' needs counts: the model is not positive and finite at every point")
        scale[:n_data] = 1 / np.sqrt(model_cur[:n_data])
        y_exp[:n_data] *= scale[:n_data]
        model_cur[:n_data] *= scale[:n_data]
    # (compute_jacobian copies params internally, so we can pass it directly)
    jac, inactive = compute_jacobian(model_func, params, x_exp, fix, confu, model_cur, Expr, NExpr,
                                     batch_eval=batch_eval)
    jac_params = _jacobian_parameters(len(params), fix, inactive)   # parameter of each row
    jac_T = jac.T
    hessian = np.matmul(jac, jac_T)           # Gauss-Newton Hessian J.J^T

    chi = y_exp - model_cur                 # chiuals

    # Gradient of 1/2 ||chi||^2 w.r.t. the free parameters: grad = jac . chi.
    # (Original computed this with an explicit i,j double loop.)
    grad = np.matmul(jac, chi) if len(jac) != 0 else np.array([], dtype=float)

    if len(grad) == 0:
        print('there is no parameters to minimize')
        stop_code = 1
    else:
        if np.max(np.abs(grad)) < eps:              # already at a stationary point
            stop_code = 1

    # Initial damping: damping = damping_scale * diagonal(hessian) (Marquardt).
    damping = damping_scale * np.diagonal(hessian).astype(float) if len(hessian) != 0 else np.array([], dtype=float)

    mu_strategy = 2        # 2 = scaled Hessian diagonal, 1 = max-diagonal * I

    # ----------------------------------------------------------------------- #
    # 2) OUTER loop: alternates the damping strategy and refreshes the Jacobian #
    # ----------------------------------------------------------------------- #
    for outer_iteration in range(0, MI):
        # Dead-end guard: strategy 1 exhausted with no usable move.
        if (stall_count == 2 and mu_strategy == 1) or (stall_count == 3 and mu_strategy == 1):
            break
        if stop_code == 1:
            break
        gain_ratio = -1
        inner_iteration = 0
        strategy_count += 1

        if True:
            if strategy_count % 2 == 0:
                # --- Strategy A: refresh Jacobian, damping = scale*diag(Hessian) --
                jac, inactive = compute_jacobian(model_func, params, x_exp, fix, confu, model_cur, Expr, NExpr,
                                                 batch_eval=batch_eval)
                jac_params = _jacobian_parameters(len(params), fix, inactive)
                if len(jac) == 0:
                    print('no move is found, J=0')
                    break
                jac_T = jac.T
                hessian = np.matmul(jac, jac_T)
                damping = damping_scale * np.diagonal(hessian).astype(float)
                grad = np.matmul(jac, chi)  # refresh gradient with the new J
                if np.max(np.abs(grad)) < eps:
                    stop_code = 1
                stall_count += 1
                mu_strategy = 2
                damping_branch = 0

            if strategy_count % 2 == 1:
                # --- Strategy B: damping = scale*max(diag(Hessian))*I (uniform) ---
                # Uses the existing jac/hessian; only the damping changes.
                damping = np.array([float(0)] * (len(jac)))
                if len(hessian) != 0:
                    damping[:] = np.amax(hessian) * damping_scale
                stall_count += 1
                mu_strategy = 1
                damping_branch = 0

        damping_growth = nu0
        stop_code = 0

        # ------------------------------------------------------------------- #
        # 3) INNER loop: try damped steps, growing damping until chi2 drops    #
        # ------------------------------------------------------------------- #
        while (gain_ratio <= 0 and stop_code == 0 and inner_iteration < MI2):
            # Solve the damped normal equations (hessian + diag(damping)).step = grad.
            if len(jac) != 0:
                step = _solve_lm(hessian, damping, grad)
            else:
                step = []

            # Active parameters (everything not fixed) — used in the relative
            # step-size convergence test below.
            free_vals = np.array([params[i] for i in range(0, len(params)) if not np.any(fix == i)],
                                 dtype=float)

            # Convergence test 2: the proposed step is tiny relative to the params.
            if norm(step) < (eps ** 2) * norm(free_vals):
                stop_code = 2

            trial = np.copy(params)

            # --- Apply the step to the free, model-affecting parameters --------
            free_i = 0
            bound_frac = np.array([float(np.inf)] * len(params))
            for i in range(0, len(params)):
                if not np.any(i == fix) and not np.any(i == inactive):
                    trial[i] = params[i] + step[free_i]
                    free_i += 1
                    if trial[i] < bounds[0][i]:
                        bound_flags[i] = -1
                    if trial[i] > bounds[1][i]:
                        bound_flags[i] = 1
            # Linear coupling: constrained params follow their source param.
            for i in range(0, len(params)):
                if np.any(i == confu[0]):
                    src = int(np.where(confu[0] == i)[0][0])
                    trial[i] = trial[int(confu[1][src])] * confu[2][src]

            # --- If the step crossed any bound, shorten it to land exactly on the
            #     closest violated bound (keeps the step feasible) --------------
            if np.any(bound_flags != 0):
                # Each parameter's fraction of the step at the bound it crossed.
                # (The original ran both lines; for a lower crossing the upper
                # line's factor 0 overwrote the fraction with 0, so with two
                # finite bounds the whole step was frozen.)
                for i in range(0, len(params)):
                    if bound_flags[i] == -1 and params[i] != trial[i]:
                        bound_frac[i] = (params[i] - bounds[0][i]) / (params[i] - trial[i])
                    if bound_flags[i] == 1 and params[i] != trial[i]:
                        bound_frac[i] = (params[i] - bounds[1][i]) / (params[i] - trial[i])
                step_clamped = step * np.min(bound_frac)   # scale step by the tightest ratio
                bound_hit = int(np.where(bound_frac == np.min(bound_frac))[0][0])
                free_i = 0
                for i in range(0, len(params)):
                    if not np.any(i == fix) and not np.any(i == inactive):
                        trial[i] = params[i] + step_clamped[free_i]
                        free_i += 1
                for i in range(0, len(params)):
                    if np.any(i == confu[0]):
                        src = int(np.where(confu[0] == i)[0][0])
                        trial[i] = trial[int(confu[1][src])] * confu[2][src]
                # On the side of the bound it crossed: the flag of bound_hit (the
                # original read it with the loop index above, i.e. the last
                # parameter's, and put an upper-bound hit on the lower bound)
                trial[bound_hit] = bounds[int((np.abs(bound_flags[bound_hit]) + np.sign(bound_flags[bound_hit])) / 2)][bound_hit]

            # --- Re-evaluate linked expressions at the trial point -------------
            params_saved = np.copy(params)
            params = trial
            for e_i in range(0, len(Expr)):
                params[NExpr[e_i]] = eval_expr(str(Expr[e_i]), params)
                if np.any(confu[1] == NExpr[e_i]):
                    matches = np.where(confu[1] == NExpr[e_i])[0]
                    for c_i in range(0, len(matches)):
                        params[int(confu[0][matches[c_i]])] = params[int(confu[1][matches[c_i]])] * confu[2][matches[c_i]]
            trial = np.copy(params)
            params = params_saved

            # --- Evaluate the model at the trial parameters --------------------
            model_trial = model_func(x_exp, trial)
            chi_trial = y_exp - model_trial         # trial chiuals

            # Predicted-reduction denominator = step . (damping*step + grad).
            predicted_drop = float(np.sum(step * (damping * step + grad))) if len(step) != 0 else float(0)

            # Gain ratio = (actual reduction) / (predicted reduction).
            # GUARD: if the model returned inf/NaN (a wild step into an invalid
            # region), reject the step (gain_ratio <= 0) instead of crashing. On
            # healthy fits model_trial is always finite, so this never fires.
            if np.all(np.isfinite(model_trial)) and poisson:
                # The Poisson fit descends on the likelihood itself: the actual
                # reduction is the deviance's (plus the penalty rows'). Its quadratic
                # model at the current point is the one of the weighted sum the
                # step was solved for, so predicted_drop is the same.
                m_cur = model_cur[:n_data] / scale[:n_data]
                m_try = model_trial[:n_data] / scale[:n_data]
                actual = (poisson_deviance(y_raw[:n_data], m_cur) + np.sum(chi[n_data:] ** 2)
                          - poisson_deviance(y_raw[:n_data], m_try) - np.sum(chi_trial[n_data:] ** 2))
                gain_ratio = float(actual / len(chi) ** 2 / np.abs(predicted_drop))
            elif np.all(np.isfinite(model_trial)):
                gain_ratio = float((norm(chi) ** 2 - norm(chi_trial) ** 2) / np.abs(predicted_drop))
            else:
                gain_ratio = -1.0

            if gain_ratio <= 0:
                # ------ Step REJECTED: increase damping and retry --------------
                if ((stall_count == 2 and mu_strategy == 1) or (stall_count == 3 and mu_strategy == 1)) and bound_hit != 0.5 and stop_code == 2:
                    # A bound-hitting parameter is the only thing left to move and
                    # it cannot improve the fit -> fix it for good.
                    fix = np.append(fix, bound_hit)
                    stall_count = 0
                damping = damping * damping_growth        # grow damping (-> steepest descent)
                damping_growth = nu0 * damping_growth     # accelerate growth next retry
                bound_flags = np.array([int(0)] * len(p0))  # forget bound flags for retry
                bound_hit = 0.5
                damping_branch = 1
            else:
                # ------ Step ACCEPTED: commit it and relax the damping ----------
                chi_old = chi
                chi = chi_trial
                grad = np.matmul(jac, chi)              # refresh gradient at accepted point
                if len(grad) != 0:
                    if np.max(np.abs(grad)) < eps:              # gradient ~ 0 -> converged
                        stop_code = 3

                # Convergence test 4: chi-square barely changed between steps.
                if accepted_any != 0:
                    dof = (n_data - len(params) + len(fix) + len(inactive))
                    if np.sum(chi_old[:n_data] ** 2 / variance) / dof \
                            - np.sum(chi[:n_data] ** 2 / variance) / dof < eps:
                        stop_code = 4
                        damping_scale = tau0
                        damping_growth = nu0

                # Commit the trial parameters and re-apply expressions on them.
                params = trial
                for e_i in range(0, len(Expr)):
                    params[NExpr[e_i]] = eval_expr(str(Expr[e_i]), params)
                    if np.any(confu[1] == NExpr[e_i]):
                        matches = np.where(confu[1] == NExpr[e_i])[0]
                        for c_i in range(0, len(matches)):
                            params[int(confu[0][matches[c_i]])] = params[int(confu[1][matches[c_i]])] * confu[2][matches[c_i]]

                model_cur = model_trial                   # accepted model becomes the base

                if poisson:
                    # The weights follow the model: 1/m at the point just accepted
                    # (docs/minimization.md, 5.1). Rows, residual and Jacobian are
                    # rescaled to them -- no model evaluation -- so the next step
                    # works with the weights of where it starts.
                    model_raw = model_cur[:n_data] / scale[:n_data]
                    if not np.all(np.isfinite(model_raw) & (model_raw > 0)):
                        raise ValueError("weights='P' needs counts: the model is not positive and finite at every point")
                    ratio = 1 / (np.sqrt(model_raw) * scale[:n_data])
                    scale[:n_data] *= ratio
                    y_exp[:n_data] *= ratio
                    model_cur[:n_data] *= ratio
                    chi[:n_data] *= ratio
                    jac[:, :n_data] *= ratio
                    jac_T = jac.T
                    hessian = np.matmul(jac, jac_T)
                    grad = np.matmul(jac, chi)

                # ---- Update the damping for the next inner iteration ----------
                if mu_strategy == 2:
                    if damping_branch == 0:
                        damping = damping_scale * np.diagonal(hessian).astype(float)
                        damping_scale = damping_scale / damping_growth
                    if damping_branch == 1:
                        damping = damping * nu0
                        damping_scale = tau0
                if mu_strategy == 1:
                    if damping_branch == 0:
                        # Nielsen's continuous damping update based on the gain ratio.
                        if np.abs(float(1 - (2 * gain_ratio - 1) ** 3)) < 1 / nu0:
                            damping = damping * np.abs(float(1 - (2 * gain_ratio - 1) ** 3))
                        else:
                            damping = damping / nu0
                    if damping_branch == 1:
                        damping = damping * nu0
                    damping_scale = tau0

                damping_growth = nu0
                gain_ratio = -1                           # force another inner attempt
                accepted_any = 1
                inner_iteration += 1
                if stop_code != 4 and stop_code != 3:
                    stall_count = 0

                if bound_hit != 0.5:
                    # A parameter reached a bound on an accepted step: fix it and
                    # break out of the inner loop so the Jacobian is rebuilt.
                    fix = np.append(fix, bound_hit)
                    fix = np.sort(fix)
                    bound_flags = np.array([int(0)] * len(p0))
                    bound_hit = 0.5
                    inner_iteration = MI2
                    stall_count = 0
                    if strategy_count % 2 == 0:
                        strategy_count += 1

    # ----------------------------------------------------------------------- #
    # 4) Final reduced chi-square                                             #
    # ----------------------------------------------------------------------- #
    dof = n_data - len(params) + len(fix) + len(inactive)
    chi2_red = np.sum(chi[:n_data] ** 2 / variance) / dof
    # Pearson's, from the unscaled model and residual at the result
    model_end = model_cur[:n_data] if scale is None else model_cur[:n_data] / scale[:n_data]
    resid_end = chi[:n_data] if scale is None else chi[:n_data] / scale[:n_data]
    chi2_pearson = np.sum(resid_end ** 2 / model_end) / dof if np.all(model_end > 0) else np.nan
    # What multiplies the covariance (step 5): Pearson's, whatever the weights
    # -- the scale of the variances the residuals show (docs/minimization.md,
    # 8.5), below 1 as well as above. Nothing does without it (a model that is
    # not counts), nor where it is 0 to the fit's own tolerance -- a spectrum
    # simulated without noise, whose errors would all be 0: the fit stops when
    # its reduced chi-square changes by less than eps (convergence test 4), so
    # one below eps cannot be told from 0.
    chi2_factor = (chi2_pearson if chi2_correction and np.isfinite(chi2_pearson) and chi2_pearson > eps
                   else 1.0)
    if poisson:
        chi2_red = poisson_deviance(y_raw[:n_data], model_end) / dof   # what it minimised

    # ----------------------------------------------------------------------- #
    # 5) Parameter covariance and 1-sigma errors                              #
    # ----------------------------------------------------------------------- #
    # The covariance of what an UNWEIGHTED fit returns (each step solves with
    # J.J^T) is the "sandwich"
    #     H^-1 . J diag(V) J^T . H^-1,    H = J.J^T (penalty rows included),
    # V = |y|+1 the variance of each data point, scaled by Pearson's reduced
    # chi-square (chi2_factor) so the errors also reflect the fit quality --
    # the covariance itself carries the factor, so errors =
    # sqrt(diag(covariance)) and an error propagated through it (a %
    # intensity, an Expression) has it too. Each variance is a
    # sum of squares, never negative, and V does not depend on the model, which
    # need not be counts (a velocity in Calibration). A weighted fit works on
    # scaled rows, V = 1, and the same formula is (J W J^T)^-1. The rows are the
    # parameters still free at the END (a bound can fix one after the last
    # Jacobian) that the data determine (_determined_rows), in index order;
    # every other parameter's error is nan, which is how the callers map rows.
    if stop_code != 1:
        free_rows = [r for r in range(len(jac_params)) if not np.any(fix == jac_params[r])]
        normal = np.matmul(jac[free_rows], jac[free_rows].T)
        kept = _determined_rows(normal)
        for r in range(len(free_rows)):
            if r not in kept:
                print('parameter No. ' + str(jac_params[free_rows[r]])
                      + ' is not determined by the data, reported as fixed')
        rows = [free_rows[r] for r in kept]
        spread = np.linalg.solve(normal[np.ix_(kept, kept)], jac[rows][:, :n_data])
        covariance = chi2_factor * np.matmul(spread * variance, spread.T)
        errors = np.array([np.nan] * len(params))
        errors[jac_params[rows]] = np.sqrt(np.diag(covariance))
    else:
        errors = np.array([0])
        covariance = np.array([[0.0]])
        print('there was no move')

    # ----------------------------------------------------------------------- #
    # 6) Boundary recheck: if a parameter was auto-fixed to a bound, nudge it  #
    #    off the bound and re-fit once to confirm the bound is really optimal. #
    # ----------------------------------------------------------------------- #
    if len(fix) != len(fix_orig) and fixCH == 0:
        print('one or more parameters fell to a boundary, rechecking')
        print('fix is ', fix)
        errors_bak = np.copy(errors)
        covariance_bak = covariance
        params_bak = np.copy(params)

        nudge_failed = 0
        newly_fixed = np.setdiff1d(fix, fix_orig)
        for i in range(0, len(newly_fixed)):
            pidx = int(newly_fixed[i])
            if params_bak[pidx] == bounds[0][pidx]:
                params[pidx] += np.maximum(10 ** -12, np.abs(params[pidx]) / 10 ** 6)
                if params[pidx] > bounds[1][pidx]:
                    nudge_failed = 1
            if params_bak[pidx] == bounds[1][pidx]:
                params[pidx] -= np.maximum(10 ** -12, np.abs(params[pidx]) / 10 ** 6)
                if params[pidx] < bounds[0][pidx]:
                    nudge_failed = 1

        if nudge_failed == 0:
            # the caller's model and data: the re-fit scales them itself
            params, errors, chi2_red, covariance, chi2_pearson = minimi_hi(
                _raw_model, x_exp, y_raw, params, fix_orig, confu, bounds,
                Expr, NExpr, np.maximum(int(MI / 2), 1), MI2, nu0, tau0, eps, 1, n_reg,
                _raw_batch, weights, chi2_correction)
            if np.all(params == params_bak):
                errors, covariance = errors_bak, covariance_bak   # a pair: same rows

    return (params, errors, chi2_red, covariance, chi2_pearson)
