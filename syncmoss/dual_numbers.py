"""Dual numbers: a value carried together with its derivatives in the fit
parameters, so that a calculation written for numbers returns its Jacobian too.

    q = seed(p, free)                      # q[k] is a Dual for every k in free
    q[7] = 0.5 * q[3]                      # a link is plain arithmetic
    q[9] = eval_expr("sqrt(p[3])", q)      # an Expression: numpy's functions take Duals
    y = model(x, q)                        # written with numpy, as for numbers
    J = jacobian(y, free)                  # dy/dp[k], one row per k in free

A Dual is its value ``v`` and ``d``, the derivatives of v in the parameters
``idx`` it depends on -- only those, so the Duals of a component carry its own
parameters, and a parameter that is not fitted stays a plain number and costs
nothing. Every operation applies the chain rule to (v, d): the derivatives
are exact, there is no step.

What takes a Dual: + - * / ** @, comparisons (of the values), indexing, and
the numpy functions in the two tables at the end of this module -- those of
expression_eval's list and the array functions the models' matrix code uses.
numpy hands them over (__array_ufunc__, __array_function__), so ``np.exp(q)``
and ``x * q`` with an array x work as written. Anything else raises: a Dual is
never turned silently into a plain number, which would drop its derivatives.

What it is for: the numbers -- parameters, links, Expressions, the positions,
widths, intensities and 2x2 matrices made of them -- and small arrays (a
distribution's axis). An array over the energies works too, but every numpy
operation on it is then repeated for every parameter: written that way a
whole spectrum is no faster than finite differences. That work, and the njit
kernels, which cannot take a Dual, have compiled twins in dual_kernels.

Where a derivative does not exist:
  * abs at 0: along each parameter the one-sided derivative |d|, what its
    forward difference gives -- the column of a parameter that starts at 0 is
    not zero, so it can leave 0;
  * sqrt and a fractional power at 0: 0, not inf. So sqrt(a**2 + b**2) has no
    derivative left at a = b = 0; np.hypot(a, b) is the form that keeps it
    (one-sided along each parameter, as abs);
  * floor, ceil, round, sign: plain numbers (their derivative is 0).
The values may be complex; the parameters are real.

A new model (models_dual, "ADDING A MODEL") that needs a numpy function not
in the tables: add it there -- a ufunc with its derivative in _UFUNCS, an
array function in _FUNCTIONS -- and its test to tests/test_dual_numbers.py.
"""
import numpy as np


class Dual:
    """The value ``v`` (a number or an array of any shape, real or complex),
    the sorted tuple ``idx`` of the parameters it depends on and its
    derivatives ``d``: d[..., j] = dv/dp[idx[j]], the shape of v plus one last
    axis."""

    def __init__(self, v, d, idx):
        self.v = np.asarray(v)
        self.d = np.asarray(d)
        self.idx = tuple(idx)
        if self.d.shape != self.v.shape + (len(self.idx),):
            raise ValueError(f"derivatives of shape {self.d.shape} for a value of shape {self.v.shape} "
                             f"in {len(self.idx)} parameters")

    def __repr__(self):
        return f"Dual({self.v!r}, derivatives in p{list(self.idx)})"

    # --- numpy hands its functions over instead of converting ---------------
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        f = _UFUNCS.get(ufunc)
        if f is None or method != '__call__' or kwargs:
            return NotImplemented
        return f(*args)

    def __array_function__(self, func, types, args, kwargs):
        f = _FUNCTIONS.get(func)
        return NotImplemented if f is None else f(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        raise TypeError("a Dual does not turn into a plain array (its derivatives would be lost); "
                        "its value is .v")

    # --- arithmetic -----------------------------------------------------------
    def __neg__(self): return _made(np.asarray(-self.v), -self.d, self.idx)
    def __pos__(self): return self
    def __abs__(self): return _abs(self)
    def __add__(self, b): return _add(self, b)
    def __radd__(self, a): return _add(a, self)
    def __sub__(self, b): return _subtract(self, b)
    def __rsub__(self, a): return _subtract(a, self)
    def __mul__(self, b): return _multiply(self, b)
    def __rmul__(self, a): return _multiply(a, self)
    def __truediv__(self, b): return _divide(self, b)
    def __rtruediv__(self, a): return _divide(a, self)
    def __pow__(self, b): return _power(self, b)
    def __rpow__(self, a): return _power(a, self)
    def __matmul__(self, b): return _matmul(self, b)
    def __rmatmul__(self, a): return _matmul(a, self)

    # --- comparisons: of the values, plain results ----------------------------
    def __lt__(self, b): return self.v < value(b)
    def __le__(self, b): return self.v <= value(b)
    def __gt__(self, b): return self.v > value(b)
    def __ge__(self, b): return self.v >= value(b)
    def __eq__(self, b): return self.v == value(b)
    def __ne__(self, b): return self.v != value(b)
    __hash__ = None
    def __bool__(self): return bool(self.v)
    def __len__(self): return len(self.v)

    # --- an array's shape and parts -------------------------------------------
    @property
    def shape(self): return self.v.shape

    @property
    def ndim(self): return self.v.ndim

    @property
    def real(self): return Dual(self.v.real, self.d.real, self.idx)

    @property
    def imag(self): return Dual(self.v.imag, self.d.imag, self.idx)

    def conj(self):
        return Dual(np.conjugate(self.v), np.conjugate(self.d), self.idx)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return Dual(self.v[key], self.d[key + (slice(None),)], self.idx)

    def __setitem__(self, key, x):
        # Into copies: the arrays of a Dual may be shared with the Duals it
        # was made from (a broadcast, a transpose, the real part).
        key = key if isinstance(key, tuple) else (key,)
        idx = _union((self, x))
        v, d = self.v.copy(), _columns(self, idx).copy()
        v[key] = value(x)
        d[key + (slice(None),)] = _columns(x, idx) if isinstance(x, Dual) else 0
        self.v, self.d, self.idx = v, d, idx

    def transpose(self, *axes):
        n = self.v.ndim
        if len(axes) == 1 and not isinstance(axes[0], (int, np.integer)):    # transpose((0, 2, 1)), transpose(None)
            axes = axes[0]
        axes = tuple(a % n for a in axes) if axes else tuple(range(n))[::-1]
        return Dual(self.v.transpose(axes), self.d.transpose(axes + (n,)), self.idx)

    def reshape(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):  # reshape((2, 3))
            shape = tuple(shape[0])
        v = self.v.reshape(shape)
        return Dual(v, self.d.reshape(v.shape + (len(self.idx),)), self.idx)

    def ravel(self):
        return self.reshape(-1)

    # --- sums over the value's axes -------------------------------------------
    def sum(self, axis=None, keepdims=False):
        axes = _axes(axis, self.v.ndim)
        return Dual(self.v.sum(axis=axes, keepdims=keepdims), self.d.sum(axis=axes, keepdims=keepdims), self.idx)

    def mean(self, axis=None, keepdims=False):
        axes = _axes(axis, self.v.ndim)
        return self.sum(axes, keepdims) / int(np.prod([self.v.shape[a] for a in axes]))

    def var(self, axis=None, keepdims=False):
        dev = self - self.mean(axis, keepdims=True)
        return (dev * dev.conj()).real.mean(axis, keepdims)

    def std(self, axis=None, keepdims=False):
        return np.sqrt(self.var(axis, keepdims))


# =============================================================================
# Making Duals and reading them
# =============================================================================
def seed(p, free):
    """The parameters *p* as a list in which every p[k], k in *free*, is a Dual
    with the derivative 1 in itself; the others stay plain numbers. Links and
    Expressions are then set by the caller, as arithmetic on this list."""
    q = [float(x) for x in p]
    for k in free:
        q[int(k)] = Dual(q[int(k)], [1.0], (int(k),))
    return q


def value(a):
    """The value of *a*, a Dual or not."""
    return a.v if isinstance(a, Dual) else a


def jacobian(y, free):
    """dy/dp[k] for the parameters *free*, one row per parameter in that order:
    zeros for a parameter y does not depend on, and for a plain y."""
    free = [int(k) for k in free]
    if not isinstance(y, Dual):
        return np.zeros((len(free),) + np.shape(y))
    rows = np.zeros((len(free),) + y.v.shape, dtype=y.d.dtype)
    for j, k in enumerate(y.idx):
        rows[free.index(k)] = y.d[..., j]
    return rows


def parameters(items):
    """The parameters any of *items* (Duals or not) depends on, sorted."""
    return _union(items)


def columns(a, idx):
    """The derivatives of *a* in the parameters *idx*, a's shape plus one last
    axis: zeros in those it does not depend on, all zeros for a plain a. For
    code that packs many values into plain arrays (a kernel's arguments)."""
    if isinstance(a, Dual):
        return _columns(a, idx)
    return np.zeros(np.shape(a) + (len(idx),))


def array(rows, dtype=None):
    """np.array(rows, dtype) for nested lists that may hold Duals (a matrix
    built from fitted angles); the plain array when they hold none."""
    if not isinstance(rows, (list, tuple)):
        return rows
    items = [array(r) for r in rows]
    idx = _union(items)
    v = np.array([value(i) for i in items], dtype=dtype)
    if not idx:
        return v
    d = np.array([_columns(i, idx) if isinstance(i, Dual) else np.zeros(np.shape(i) + (len(idx),))
                  for i in items])
    return Dual(v, d, idx)


# =============================================================================
# The chain rule
# =============================================================================
def _made(v, d, idx):
    """Dual(v, d, idx) for arrays that are already right: every operation
    ends here, so it skips the conversions and the shape check."""
    a = Dual.__new__(Dual)
    a.v, a.d, a.idx = v, d, idx
    return a


def _union(args):
    """The parameters any of *args* depends on, sorted."""
    idx = ()
    for a in args:
        if isinstance(a, Dual) and a.idx != idx:
            idx = tuple(sorted(set(idx + a.idx))) if idx else a.idx
    return idx


def _columns(a, idx):
    """The derivatives of the Dual *a* in the parameters *idx*: zeros in those
    it does not depend on."""
    if a.idx == idx:
        return a.d
    d = np.zeros(a.v.shape + (len(idx),), dtype=a.d.dtype)
    d[..., [idx.index(k) for k in a.idx]] = a.d
    return d


def chain(v, partials, args):
    """f(*args) as a Dual, from its value *v* and *partials*, its derivative in
    each argument (None for 1; not read where the argument is not a Dual).
    The plain v when no argument is a Dual. A kernel's twin, which returns the
    value and these derivatives as plain arrays, is made a Dual this way."""
    idx = _union(args)
    if not idx:
        return v
    v = np.asarray(v)
    d = None
    for part, a in zip(partials, args):
        if isinstance(a, Dual):
            term = _columns(a, idx)
            if part is not None:
                part = np.asarray(part)
                term = (part[..., None] if part.ndim else part) * term
            d = term if d is None else d + term
    shape = v.shape + (len(idx),)
    return _made(v, d if d.shape == shape else np.broadcast_to(d, shape), idx)


def _add(a, b):
    return chain(value(a) + value(b), (None, None), (a, b))


def _subtract(a, b):
    return chain(value(a) - value(b), (None, -1.0), (a, b))


def _multiply(a, b):
    va, vb = value(a), value(b)
    return chain(va * vb, (vb, va), (a, b))


def _divide(a, b):
    va, vb = value(a), value(b)
    v = va / vb
    return chain(v, (1 / vb, -v / vb), (a, b))


def _power(a, b):
    va, vb = value(a), value(b)
    v = va ** vb
    in_a = in_b = None
    with np.errstate(divide='ignore', invalid='ignore'):
        if isinstance(a, Dual):
            in_a = vb * va ** (vb - 1)
            in_a = np.where((va == 0) & ~np.isfinite(in_a), 0, in_a)     # a fractional power at 0
        if isinstance(b, Dual):
            in_b = np.where(va == 0, 0, v * np.log(va))
    return chain(v, (in_a, in_b), (a, b))


def _matmul(a, b):
    """Stacks of matrices, as np.matmul: d(AB) = dA B + A dB, the derivative
    axis taken to the front for the products (ahead of the stacks' own axes,
    so that a single matrix still multiplies a stack)."""
    va, vb = np.asarray(value(a)), np.asarray(value(b))
    idx = _union((a, b))
    ndim = max(va.ndim, vb.ndim)

    def front(x, v):
        return np.moveaxis(_columns(x, idx), -1, 0).reshape((len(idx),) + (1,) * (ndim - v.ndim) + v.shape)
    d = 0
    if isinstance(a, Dual):
        d = d + np.matmul(front(a, va), vb)
    if isinstance(b, Dual):
        d = d + np.matmul(va, front(b, vb))
    return Dual(np.matmul(va, vb), np.moveaxis(d, 0, -1), idx)


def _abs(a):
    v = np.abs(a.v)
    if np.iscomplexobj(a.v):
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.real((np.conjugate(a.v) / v)[..., None] * a.d)       # |z|' = Re(conj(z) z')/|z|
    else:
        d = np.sign(a.v)[..., None] * a.d
    # At 0 the derivative along a parameter is |d|, from |h d|/h: one-sided
    return Dual(v, np.where((v == 0)[..., None], np.abs(a.d), d), a.idx)


def _hypot(a, b):
    va, vb = value(a), value(b)
    v = np.hypot(va, vb)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = chain(v, (va / v, vb / v), (a, b))
    if np.any(v == 0):
        # At (0, 0) the derivative along a parameter is hypot(da, db), from
        # hypot(h da, h db)/h: one-sided, as abs at 0
        da = _columns(a, out.idx) if isinstance(a, Dual) else 0.0
        db = _columns(b, out.idx) if isinstance(b, Dual) else 0.0
        out.d = np.where((v == 0)[..., None], np.hypot(da, db), out.d)
    return out


def where(c, a, b):
    """np.where(c, a, b) for a, b that may be Duals."""
    c = value(c)
    v = np.where(c, value(a), value(b))
    idx = _union((a, b))
    if not idx:
        return v
    d = np.where(np.asarray(c)[..., None],
                 _columns(a, idx) if isinstance(a, Dual) else 0.0,
                 _columns(b, idx) if isinstance(b, Dual) else 0.0)
    return Dual(v, np.broadcast_to(d, v.shape + (len(idx),)), idx)


def _maximum(a, b):
    return where(value(a) >= value(b), a, b)


def _minimum(a, b):
    return where(value(a) <= value(b), a, b)


def _function(f, derivative):
    """f for a Dual: *derivative*(v, f(v)) is f'(v)."""
    def of_dual(a):
        v = np.asarray(f(a.v))
        slope = np.asarray(derivative(a.v, v))
        return _made(v, (slope[..., None] if slope.ndim else slope) * a.d, a.idx)
    return of_dual


def _of_values(f):
    """f of the values, a plain result: a comparison, or a function whose
    derivative is 0."""
    def plain(*args):
        return f(*[value(a) for a in args])
    return plain


def _sqrt_derivative(v, f):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(f == 0, 0, 0.5 / f)


# =============================================================================
# The array functions
# =============================================================================
def _axes(axis, ndim):
    """*axis* of a sum (None: all of them) as a tuple of the value's axes, a
    negative one counted from the value's last."""
    if axis is None:
        return tuple(range(ndim))
    return tuple(int(a) % ndim for a in np.atleast_1d(axis))


def _zeros_like(a, dtype=None):
    v = np.zeros_like(a.v, dtype=dtype)
    return Dual(v, np.zeros(a.d.shape, dtype=v.dtype), a.idx)


def _concatenate(parts, axis=0):
    idx = _union(parts)
    axis = axis % np.ndim(value(parts[0]))
    return Dual(np.concatenate([value(a) for a in parts], axis=axis),
                np.concatenate([columns(a, idx) for a in parts], axis=axis), idx)


def _linspace(start, stop, num=50):
    t = np.linspace(0.0, 1.0, num)
    return chain(np.linspace(value(start), value(stop), num), (1 - t, t), (start, stop))


# =============================================================================
# What numpy hands over
# =============================================================================
_UFUNCS = {
    np.add: _add, np.subtract: _subtract, np.multiply: _multiply, np.true_divide: _divide,
    np.power: _power, np.float_power: _power, np.matmul: _matmul, np.hypot: _hypot,
    np.maximum: _maximum, np.minimum: _minimum,
    np.negative: lambda a: -a,
    np.positive: lambda a: a,
    np.absolute: _abs,
    np.conjugate: lambda a: a.conj(),
    np.square: lambda a: _multiply(a, a),
    np.sqrt: _function(np.sqrt, _sqrt_derivative),
    np.exp: _function(np.exp, lambda v, f: f),
    np.log: _function(np.log, lambda v, f: 1 / v),
    np.log10: _function(np.log10, lambda v, f: 1 / (v * np.log(10))),
    np.sin: _function(np.sin, lambda v, f: np.cos(v)),
    np.cos: _function(np.cos, lambda v, f: -np.sin(v)),
    np.tan: _function(np.tan, lambda v, f: 1 + f * f),
    np.arcsin: _function(np.arcsin, lambda v, f: 1 / np.sqrt(1 - v * v)),
    np.arccos: _function(np.arccos, lambda v, f: -1 / np.sqrt(1 - v * v)),
    np.arctan: _function(np.arctan, lambda v, f: 1 / (1 + v * v)),
    np.sinh: _function(np.sinh, lambda v, f: np.cosh(v)),
    np.cosh: _function(np.cosh, lambda v, f: np.sinh(v)),
    np.tanh: _function(np.tanh, lambda v, f: 1 - f * f),
    np.arcsinh: _function(np.arcsinh, lambda v, f: 1 / np.sqrt(v * v + 1)),
    np.arccosh: _function(np.arccosh, lambda v, f: 1 / np.sqrt(v * v - 1)),
    np.arctanh: _function(np.arctanh, lambda v, f: 1 / (1 - v * v)),
}
for _f in (np.floor, np.ceil, np.rint, np.trunc, np.sign, np.isfinite, np.isnan, np.isinf,
           np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal):
    _UFUNCS[_f] = _of_values(_f)

_FUNCTIONS = {
    np.sum: lambda a, axis=None, keepdims=False: a.sum(axis, keepdims),
    np.mean: lambda a, axis=None, keepdims=False: a.mean(axis, keepdims),
    np.var: lambda a, axis=None, keepdims=False: a.var(axis, keepdims),
    np.std: lambda a, axis=None, keepdims=False: a.std(axis, keepdims),
    np.real: lambda a: a.real,
    np.imag: lambda a: a.imag,
    np.transpose: lambda a, axes=None: a.transpose(axes),
    np.reshape: lambda a, shape: a.reshape(shape),
    np.ravel: lambda a: a.ravel(),
    np.shape: lambda a: a.v.shape,
    np.round: lambda a, decimals=0: np.round(a.v, decimals),
    np.where: where,
    np.empty_like: _zeros_like,
    np.zeros_like: _zeros_like,
    np.linspace: _linspace,
    np.concatenate: _concatenate,
}
