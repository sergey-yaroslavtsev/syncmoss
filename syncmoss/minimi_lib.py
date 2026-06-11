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
              MI=, MI2=, nu0=, tau0=, eps=, fixCH=)
        -> (params, errors, chi2_red, covariance)
Damped Gauss-Newton (Levenberg-Marquardt) fit of ``model_func(x_exp, params)`` to
``y_exp``, with support for fixed parameters (``fix``), linear parameter coupling
(``confu``), box ``bounds`` and linked expressions (``Expr`` evaluated into the
``NExpr`` indices).

Implementation notes:
  * The numba linear-algebra kernels (``solution``, ``sim_inv``) are
    ``@njit(cache=True)``, so their one-time compilation is cached on disk and
    reused across launches / multiprocessing workers; ``warmup()`` lets the
    application trigger that compile at a predictable moment.
  * The gradient, the damped normal-equation assembly, the chi-square sums and the
    Jacobian assembly are vectorised; the parameter covariance is formed by the
    broadcast ``(jac / model) . jac_T`` rather than a dense (N_data x N_data)
    weight matrix.
  * A trial step whose model output is non-finite is rejected (damping increased)
    rather than crashing the fit.
  * Variable names are descriptive; the textbook LM symbols map as
    mu->damping, tau->damping_scale, nu->damping_growth, rho->gain_ratio,
    chi->residual vector (y - model), J->jac, A->hessian, g->grad.
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
from numba import njit

# The ``from numpy import (...)`` block above (same set as syncmoss_main.py) exists
# ONLY so that user-supplied ``Expr`` strings evaluated by ``_eval_expr`` below can
# use bare numpy functions, e.g. "sqrt(p[3])". Our own code always calls numpy
# explicitly as ``np.<func>``.


def _eval_expr(expr, p):
    """Evaluate a user parameter-linking expression string (the ``Expr`` feature).

    ``p`` is the current parameter array, so the string may reference ``p[i]``;
    bare numpy functions (``sqrt``, ``exp``, ...) resolve through this module's
    globals (the imports above). The expression text comes from the user, so this
    is the one place where bare (non-``np.``) numpy names are intended."""
    return eval(expr, globals(), {"p": p})


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


@njit(cache=True)
def sim_inv(matrix):
    """Matrix inverse by Gauss-Jordan reduction of ``[matrix | I]``. Used for the
    parameter covariance. (Original kernel; faster than numpy.linalg.inv for the
    small matrices here, with a numpy fallback at the call site.)"""
    reduced = np.copy(matrix)
    pivot = 0
    identity = np.identity(len(matrix))
    reduced = np.concatenate((reduced, identity), axis=-1)
    for col in range(0, len(matrix)):
        nz = np.nonzero(reduced[pivot:, col])[0]
        if nz.size == 0:
            continue
        row = pivot + nz[0]
        tmp = reduced[pivot, :]
        reduced[pivot, :] = reduced[row, :]
        reduced[row, :] = tmp
        reduced[pivot, :] = reduced[pivot, :] / reduced[pivot, col]
        nz = np.nonzero(reduced[:, col])[0].flatten()
        nz = np.delete(nz, pivot)
        for r in range(0, len(nz)):
            reduced[nz[r], :] -= reduced[nz[r], col] * reduced[pivot, :]
        pivot += 1
        if pivot == reduced.shape[0]:
            break
    return reduced[:, -len(matrix):]


def warmup():
    """Trigger (or load from cache) the JIT compilation of the kernels above.

    Call this ONCE at application start (e.g. while a splash screen is up) so the
    one-time numba compile happens at a predictable moment instead of inside the
    user's first fit. With ``cache=True`` the compiled code is reused on every
    later launch and in every pool worker, so this becomes a fast cache-load
    rather than a recompile. Safe to call multiple times.
    """
    # Use a dense, well-conditioned system (like a real LM/covariance matrix);
    # a purely diagonal matrix would hit an edge case in sim_inv's pivoting.
    aug = np.array([[2.0, 1.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    solution(np.copy(aug))
    sim_inv(np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64))


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


# =============================================================================
# Jacobian by forward finite differences
# =============================================================================
def compute_jacobian(model_func, params, x_exp, fix, confu, model_base,
                     Expr, NExpr, verbose=False):
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

    Returns ``(jac, inactive)``:
      * ``jac``  : array (n_effective_free, n_data), one row per parameter that
                   actually moves the model.
      * ``inactive`` : indices of parameters whose column came out all-zero (they do
                   not affect the model); the caller excludes them from steps.
    """
    n_data = len(x_exp)
    jac_rows = []        # Jacobian rows for parameters that move the model
    inactive = []            # indices of "inactive" parameters (all-zero column)

    for k in range(0, len(params)):
        if np.any(fix == k):
            continue

        bumped = np.copy(params)
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
        perturbation = np.maximum(10 ** -12, np.abs(bumped[k]) / 10 ** 6)
        bumped[k] = bumped[k] + perturbation

        # Re-evaluate linked expressions after the bump and propagate their change
        # through any linear constraints attached to the expression target.
        for e_i in range(0, len(Expr)):
            bumped[NExpr[e_i]] = _eval_expr(str(Expr[e_i]), bumped)
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

        # One model evaluation -> one Jacobian column (forward difference).
        model_up = model_func(x_exp, bumped)
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
# Levenberg-Marquardt least-squares fit  (public API — unchanged)
# =============================================================================
def minimi_hi(model_func, x_exp, y_exp, p0, fix=np.array([], dtype=int),
              confu=np.array([[-1], [-1], [0]], dtype=float),
              bounds=np.array([[], []], dtype=float), Expr=[],
              NExpr=np.array([], dtype=int), MI=10, MI2=20, nu0=2.618,
              tau0=0.001, eps=10 ** -10, fixCH=0):
    """Levenberg-Marquardt least-squares fit of ``model_func(x_exp, params)`` to ``y_exp``.

    Optimised twin of the previous ``minimi_lib_old.minimi_hi`` — identical signature
    (keyword names unchanged: ``fix, confu, bounds, Expr, NExpr, MI, MI2, nu0,
    tau0, eps, fixCH``), identical algorithm and return value, faster internals
    and a non-finite-result guard (see the module docstring). Supports fixed
    parameters (``fix``), linear parameter coupling (``confu``), box ``bounds``
    and linked expressions (``Expr`` evaluated into the ``NExpr`` indices).
    ``MI`` / ``MI2`` cap the outer/inner iterations; ``eps`` is the tolerance.

    Returns:
        tuple ``(params, errors, chi2_red, covariance)`` - fitted parameters,
        their 1-sigma errors (``nan`` for fixed/inactive parameters), the reduced
        chi-square, and the parameter covariance matrix.
    """

    # ----------------------------------------------------------------------- #
    # 0) Set-up: coerce inputs, initialise the LM state machine                #
    # ----------------------------------------------------------------------- #
    x_exp = np.array(x_exp, dtype=float)
    y_exp = np.array(y_exp, dtype=float)
    fix_orig = np.copy(fix)                   # remember the user's original fixes

    # Some SYNCmoss models (e.g. models.TI) return an ``object``-dtype array. The
    # numba kernels need real float64, so wrap the user model once to coerce every
    # output to float64 (lossless — the objects are Python floats). The wrapper
    # forwards to the raw model so any caller-side call counter still sees exactly
    # one call per evaluation.
    _raw_model = model_func

    def model_func(_x, _p):
        return np.asarray(_raw_model(_x, _p), dtype=float)

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
    # (compute_jacobian copies params internally, so we can pass it directly)
    jac, inactive = compute_jacobian(model_func, params, x_exp, fix, confu, model_cur, Expr, NExpr)
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
                jac, inactive = compute_jacobian(model_func, params, x_exp, fix, confu, model_cur, Expr, NExpr)
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
                for i in range(0, len(params)):
                    if bounds[0][i] != -np.inf and params[i] != trial[i] and bound_flags[i] != 0:
                        bound_frac[i] = (params[i] - bounds[0][i]) / (params[i] - trial[i]) * (np.abs(bound_flags[i]) - np.sign(bound_flags[i])) / 2
                    if bounds[1][i] != np.inf and params[i] != trial[i] and bound_flags[i] != 0:
                        bound_frac[i] = (params[i] - bounds[1][i]) / (params[i] - trial[i]) * (np.abs(bound_flags[i]) + np.sign(bound_flags[i])) / 2
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
                trial[bound_hit] = bounds[int((np.abs(bound_flags[i]) + np.sign(bound_flags[i])) / 2)][bound_hit]

            # --- Re-evaluate linked expressions at the trial point -------------
            params_saved = np.copy(params)
            params = trial
            for e_i in range(0, len(Expr)):
                params[NExpr[e_i]] = _eval_expr(str(Expr[e_i]), params)
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
            if np.all(np.isfinite(model_trial)):
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
                    dof = (len(y_exp) - len(params) + len(fix) + len(inactive))
                    if np.sum(chi_old ** 2 / (np.abs(y_exp) + 1)) / dof \
                            - np.sum(chi ** 2 / (np.abs(y_exp) + 1)) / dof < eps:
                        stop_code = 4
                        damping_scale = tau0
                        damping_growth = nu0

                # Commit the trial parameters and re-apply expressions on them.
                params = trial
                for e_i in range(0, len(Expr)):
                    params[NExpr[e_i]] = _eval_expr(str(Expr[e_i]), params)
                    if np.any(confu[1] == NExpr[e_i]):
                        matches = np.where(confu[1] == NExpr[e_i])[0]
                        for c_i in range(0, len(matches)):
                            params[int(confu[0][matches[c_i]])] = params[int(confu[1][matches[c_i]])] * confu[2][matches[c_i]]

                model_cur = model_trial                   # accepted model becomes the base

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
    chi2_red = np.sum(chi ** 2 / (np.abs(y_exp) + 1)) / (len(y_exp) - len(params) + len(fix) + len(inactive))

    # ----------------------------------------------------------------------- #
    # 5) Parameter covariance and 1-sigma errors                              #
    # ----------------------------------------------------------------------- #
    # The (unscaled) curvature in data space is jac . diag(1/model) . jac^T.
    # Original materialised the full N_data x N_data diagonal diag(1/model) and did
    # two dense matmuls; (jac / model_trial) broadcasts the same weighting at
    # O(N_data*N_param) cost and is algebraically identical.
    if stop_code != 1:
        curvature = np.matmul(jac / model_trial, jac_T)
        curvature = np.ascontiguousarray(curvature, dtype=np.float64)
        try:
            try:
                covariance = sim_inv(curvature)          # fast for small matrices
            except Exception:
                print('error in fast calculation of inverse matrix')
                covariance = np.linalg.inv(curvature)    # robust fallback (two spectra)
            errors = np.array([float(0)] * len(params))
            free_i = 0
            for i in range(0, len(params)):
                if not np.any(fix == i) and not np.any(inactive == i):
                    errors[i] = np.sqrt(np.diag(covariance)[free_i])
                    free_i += 1
                else:
                    errors[i] = None                     # fixed/inactive -> nan
        except np.linalg.LinAlgError:
            print('numpy error! parameters errors could not be found')
            errors = np.array([float(0)] * len(params))
        # Scale errors by the reduced chi-square (so they reflect fit quality).
        errors = np.sqrt(np.sum(chi ** 2 / (np.abs(y_exp) + 1))
                         / (len(y_exp) - len(params) + len(fix) + len(inactive))) * errors
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
            params, errors, chi2_red, covariance = minimi_hi(
                model_func, x_exp, y_exp, params, fix_orig, confu, bounds,
                Expr, NExpr, np.maximum(int(MI / 2), 1), MI2, nu0, tau0, eps, 1)
            if np.all(params == params_bak):
                errors = errors_bak

    return (params, errors, chi2_red, covariance)
