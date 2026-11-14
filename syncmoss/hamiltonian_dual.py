"""models.Ham_mosaic for arguments that may be Duals (dual_numbers): the eight
lines of the Hamiltonian family -- their 2x2 matrices and positions -- with
their exact derivatives.

It is the closed form of models._ham_tensor_core and models.Ham_mosaic
(K. R. Szymanski, J. Synchrotron Rad. 33, 931 (2026); docs/Hamiltonian_analytical.md),
formula by formula, written with numpy so that it takes Duals. It runs once
per spectrum (per copy of a distribution), on numbers and 3x3 arrays.

One thing is NOT carried through the formulas: the four excited levels. They
are the roots of the quartic lambda^4 + p lambda^2 + q lambda + r, and the
closed form of its roots has no derivative where two of them meet (B = 0
always; crossings). So the levels are found on the VALUES of p, q, r, as the
kernel finds them, and their derivatives come from the quartic itself:

  * a level that stands alone: d(lambda) = -(lambda^2 dp + lambda dq + dr) / (4 lambda^3 + 2 p lambda + q);
  * two close levels: only their sum and product are used, as the kernel's
    pair formulas use only their mean and half-distance. With the quartic as
    (lambda^2 - s lambda + a)(lambda^2 + s lambda + b),
        p = a + b - s^2,   q = s (a - b),   r = a b,
    the derivatives of s, a, b solve a 3x3 system that stays regular when the
    two levels of a pair meet.

Where the two levels of a pair COINCIDE (to float64, as the kernel decides it:
no field; a crossing) the two lines stand at one position, and separately
they have no derivative in what splits them: their half-distance rho is
sqrt(rho^2) with rho^2 at its minimum 0. Their SUM has one. With the pair's
lines I(-+) = both/2 -+ m1/(2 rho) at sc (lb -+ rho), and V a line shape,

    I(-) V(E - sc (lb - rho)) + I(+) V(E - sc (lb + rho))
        = both V(E - sc lb) + sc m1 dV/dpos (E - sc lb) + O(rho^2),

and both and m1 are smooth (m1 = 0 where the levels meet). So a coincident
pair is returned as these two terms instead of its two lines: the line `both`
at the pair's position, and its FIRST MOMENT sc m1 -- of the value 0, with
its derivatives -- to be multiplied by the shape's derivative in the position
(dual_kernels.SLOPE). The spectrum is the kernel's, and its derivatives are
exact there too. Ham_mosaic says which lines are moments.

With Q = H = 0 all four levels coincide and the kernel returns one line: its
derivatives in Q and H are not carried (models_dual.fit_jacobian differences
a row that is all zeros).

Just above the coincidence the two lines' difference loses digits to
rounding -- the slope in the field is good to 4e-10 down to 1e-10 T, to 1e-6
at 1e-12 T, to 2e-3 at 1e-14 T (docs/dual_numbers/hamiltonian_weak_field.py).
"""
import numpy as np

import syncmoss.dual_numbers as dn
from syncmoss.constants import E0_J, c, ggr, gex, mun

_I3 = np.eye(3)
_FLOAT_EPS = np.finfo(float).eps


def _levels(p, q, r):
    """The roots of lambda^4 + p lambda^2 + q lambda + r, ascending: Appendix A
    of the paper as models._ham_tensor_core has it, on numbers."""
    Z = np.sqrt(p * p + 12 * r)
    Y = min(1.0, max(-1.0, (2 * p ** 3 + 27 * q * q - 72 * p * r) / (2 * Z ** 3)))
    X = -p / 6 + Z / 6 * np.cos(np.arccos(Y) / 3)
    xi1 = np.sqrt(X)
    xi2 = q / (4 * xi1)
    xi3 = -(X + p / 2)
    xi4 = np.sqrt(xi3 + xi2) if xi3 + xi2 > 0.0 else 0.0
    xi5 = np.sqrt(xi3 - xi2) if xi3 - xi2 > 0.0 else 0.0
    return np.sort(np.array([xi1 + xi5, xi1 - xi5, -xi1 + xi4, -xi1 - xi4]))


def _alone(level, p, q, r):
    """A level that is a simple root, with its derivatives."""
    slope = 4 * level ** 3 + 2 * dn.value(p) * level + dn.value(q)
    return dn.chain(level, (-level ** 2 / slope, -level / slope, -1.0 / slope), (p, q, r))


def _factors(pair, others, p, q, r):
    """The sum s and the product a of the levels *pair* and the product b of
    the *others* (their sum is -s), with their derivatives."""
    s, a, b = pair[0] + pair[1], pair[0] * pair[1], others[0] * others[1]
    inverse = np.linalg.inv(np.array([[-2 * s, 1.0, 1.0], [a - b, s, -s], [0.0, b, a]]))
    return [dn.chain(x, inverse[i], (p, q, r)) for i, x in enumerate((s, a, b))]


def _det4(A):
    """The determinant of a 4x4 matrix given as rows of numbers or Duals:
    models._ham_det4's elimination with partial pivoting."""
    A = [list(row) for row in A]
    det = 1.0
    for col in range(4):
        piv = max(range(col, 4), key=lambda row: abs(dn.value(A[row][col])))
        if abs(dn.value(A[piv][col])) == 0.0:
            return 0.0
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            det = -det
        det = det * A[col][col]
        for row in range(col + 1, 4):
            fct = A[row][col] / A[col][col]
            for cc in range(col + 1, 4):
                A[row][cc] = A[row][cc] - fct * A[col][cc]
    return det


def _outer(a, b):
    return a[:, None] * b[None, :]


def _stacked(parts):
    """Per-line results (8 arrays of one shape, numbers or Duals) as one array."""
    return dn.array(list(parts))


def _tensor_core(Q, Hhf, etto, phi, tet):
    """models._ham_tensor_core: It (8, 3, 3), Gc (8, 3), S (8), and which of
    the eight are not lines but the first moments of a coincident pair (8,)."""
    b = Hhf * gex * mun * c / E0_J
    bg = Hhf * ggr * mun * c / E0_J
    sc = max(abs(dn.value(Q)), abs(dn.value(b)))
    sc = 2.0 ** np.floor(np.log2(sc)) if sc > 0.0 else 1.0
    Q = Q / sc
    b = b / sc
    t = tet / 180 * np.pi
    f = phi / 180 * np.pi
    st, ct = np.sin(t), np.cos(t)
    m = dn.array([st * np.cos(f), st * np.sin(f), ct])
    Vd = dn.array([-(1 - etto) / 2 * Q, -(1 + etto) / 2 * Q, Q])
    Vm = Vd * m
    V2m = Vd * Vm
    sm = np.sum(m * Vm)
    Sm = np.sum(m * V2m)
    D2 = Q * Q * (3 + etto * etto) / 12
    p = -(5 * b * b + D2) / 2
    q = -b * b * sm
    r = (9 * b ** 4 + D2 * D2 + b * b * (6 * D2 - 4 * Sm)) / 16
    if dn.value(p) == 0.0:                  # Q = H = 0: one line
        return np.array([_I3 / 8] * 8), np.zeros((8, 3)), np.zeros(8), np.zeros(8, dtype=bool)

    lam = _levels(float(dn.value(p)), float(dn.value(q)), float(dn.value(r)))
    g01, g12, g23 = lam[1] - lam[0], lam[2] - lam[1], lam[3] - lam[2]
    pairs = [(1, 2, 0, 3)] if g12 < g01 and g12 <= g23 else [(0, 1, 2, 3), (2, 3, 0, 1)]

    # the matrix of eq. (5), for the half-distance of a pair: rho^2 = -det(lb - H)/R(lb)
    eip = np.cos(f) - 1j * np.sin(f)
    H = [[0.0] * 4 for _ in range(4)]
    H[0][0] = Q / 4 - 1.5 * b * ct
    H[1][1] = -Q / 4 - 0.5 * b * ct
    H[2][2] = -Q / 4 + 0.5 * b * ct
    H[3][3] = Q / 4 + 1.5 * b * ct
    H[0][1] = -np.sqrt(3) / 2 * b * st * eip
    H[1][2] = -b * st * eip
    H[2][3] = H[0][1]
    H[0][2] = Q * etto / (4 * np.sqrt(3))
    H[1][3] = H[0][2]
    for i in range(4):
        for j in range(i):
            H[i][j] = np.conjugate(H[j][i])
    Om = max(abs(lam[0]), abs(lam[3]))

    # per pair: its mean lb and half-distance squared rho2, and of the other
    # two levels their sum and product (all the pair's lines read of them)
    close = []
    for lo, hi, o1, o2 in pairs:
        s, a, bq = _factors((lam[lo], lam[hi]), (lam[o1], lam[o2]), p, q, r)
        lb = s / 2
        beside = lb * lb + s * lb + bq                          # (lb - lam[o1]) (lb - lam[o2])
        rho2 = -np.real(_det4([[(lb if i == j else 0.0) - H[i][j] for j in range(4)] for i in range(4)])) / beside
        if dn.value(rho2) <= (_FLOAT_EPS * Om) ** 2:            # closer than float64 resolves: coincident
            rho2 = 0.0
        close.append((lo, hi, lb, rho2, -s, beside))
    alone = []
    if len(pairs) == 1:                                         # the middle pair; the outer levels alone
        low, high = _alone(lam[0], p, q, r), _alone(lam[3], p, q, r)
        alone = [(0, low, high, s, a), (3, high, low, s, a)]    # the level, the other one, the pair's sum and product

    It, Gc, S = [None] * 8, [None] * 8, [None] * 8
    moment = np.zeros(8, dtype=bool)
    for jb in range(2):
        beta = 1.0 - 2.0 * jb
        if dn.value(bg) > 0.0:              # the lower ground level first
            beta = -beta
        u = Vm + 3 * b * beta * m
        CI = [(2 * b * beta * (_I3 * (Vd * Vd)[None, :] - _outer(u, u))
               + _I3 * ((3 * b * b + D2) * Vd + 2 * b * beta * (Sm - 3 * D2))[None, :]) / 32,
              (-4 * b * beta * (_outer(u, m) + _outer(m, u))
               + _I3 * (-4 * b * beta * Vd + 8 * b * beta * sm - 4 * (3 * b * b + D2))[None, :]) / 32,
              (-8 * b * beta * _outer(m, m) + _I3 * (16 * b * beta - 4 * Vd)[None, :]) / 32,
              _I3 / 2]
        CG = [(beta * (4 * b * b + p) * Vm + 3 * b * (b * b + p) * m + b * V2m) / 16,
              (4 * b * Vm + 2 * beta * (4 * b * b + p) * m) / 16,
              (2 * beta * Vm + 10 * b * m) / 16,
              beta * m / 4]
        k0 = 4 * jb
        for level, L, other, s_pair, a_pair in alone:           # eq. (17) at a level that stands alone
            w = (L - other) * (L * L - s_pair * L + a_pair)
            It[k0 + level] = (CI[0] + L * (CI[1] + L * (CI[2] + L * CI[3]))) / w
            Gc[k0 + level] = -(CG[0] + L * (CG[1] + L * (CG[2] + L * CG[3]))) / w
            S[k0 + level] = sc * L - beta * bg / 2
        for lo, hi, lb, rho2, s_others, beside in close:
            # the pair's two lines from their sum and rho times their difference
            rho = np.sqrt(rho2)
            Re = beside + rho2
            Ro = 2.0 * lb - s_others
            den = Re * Re - rho2 * Ro * Ro
            e2 = lb * lb + rho2
            e3 = lb * (lb * lb + 3.0 * rho2)
            o2c = 3.0 * lb * lb + rho2
            coincident = dn.value(rho) == 0.0
            for C, out, sign in ((CI, It, 1.0), (CG, Gc, -1.0)):
                Ne = C[0] + C[1] * lb + C[2] * e2 + C[3] * e3
                No = C[1] + 2.0 * C[2] * lb + C[3] * o2c
                both = (No * Re - Ne * Ro) / den
                m1 = (Ne * Re - rho2 * No * Ro) / den
                if coincident:              # the pair's sum, and its first moment: 0 with its derivatives
                    out[k0 + lo] = sign * both
                    out[k0 + hi] = sign * sc * (m1 - dn.value(m1))
                else:
                    out[k0 + lo] = sign * (0.5 * both - 0.5 / rho * m1)
                    out[k0 + hi] = sign * (0.5 * both + 0.5 / rho * m1)
            moment[k0 + hi] = coincident
            S[k0 + lo] = sc * (lb - rho) - beta * bg / 2
            S[k0 + hi] = sc * (lb + rho) - beta * bg / 2
    return _stacked(It), _stacked(Gc), _stacked(S), moment


def _R0_rows(tetr, phir, alfak, cms):
    """models._R0_rows: the lab axes (e1, e2, k) as rows, in the EFG's axes."""
    t = tetr / 180 * np.pi
    ph = phir / 180 * np.pi
    ak = alfak / 180 * np.pi
    nr = dn.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)])
    that = dn.array([np.cos(t) * np.cos(ph), np.cos(t) * np.sin(ph), -np.sin(t)])
    phat = dn.array([-np.sin(ph), np.cos(ph), 0.0 * ph])
    if cms:
        return dn.array([that, phat, nr])
    return dn.array([nr, np.sin(ak) * that - np.cos(ak) * phat, np.cos(ak) * that + np.sin(ak) * phat])


def _cross(a, b):
    return dn.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _rank2_scale(X, axis, lam0, lam1, lam2):
    """models._rank2_scale for the eight tensors X (8, 3, 3) at once: their
    m = 0, +-1, +-2 parts about *axis* scaled by lam0, lam1, lam2."""
    z = axis / np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    t = np.array([1.0, 0.0, 0.0]) if abs(dn.value(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _cross(t, z)
    x = x / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    y = _cross(z, x)
    Lt = dn.array([x, y, z])                    # the rows x, y, z: L's transpose
    L = np.transpose(Lt)
    Xl = np.matmul(np.matmul(Lt, X), L)
    a = Xl[:, 2, 2]
    b = 0.5 * (Xl[:, 0, 0] - Xl[:, 1, 1])
    off, xz, yz = lam2 * Xl[:, 0, 1], lam1 * Xl[:, 0, 2], lam1 * Xl[:, 1, 2]
    Xp = dn.array([[-0.5 * lam0 * a + lam2 * b, off, xz],
                   [off, -0.5 * lam0 * a - lam2 * b, yz],
                   [xz, yz, lam0 * a]])
    return np.matmul(np.matmul(L, np.transpose(Xp, (2, 0, 1))), Lt)


def Ham_mosaic(Q, Hhf, etto, phi, tet, tetr, phir, alfak, Atex, Am, Ah, cms):
    """models.Ham_mosaic: P (8, 2, 2), the lines' matrices in (e1, e2), and
    S (8), their positions in mm/s -- and moment (8,): True for what is not a
    line but the first moment of a coincident pair, in mm/s, whose shape is
    the line shape's derivative in the position (the pair's sum is the line
    before it). No level coincides: the kernel's eight lines."""
    It, Gc, S, moment = _tensor_core(Q, Hhf, etto, phi, tet)
    R0 = _R0_rows(tetr, phir, alfak, cms)
    fl = np.array([0.0, 0.0, 1.0]) if cms else np.array([1.0, 0.0, 0.0])    # the lab reference axis
    n0 = R0[2] if cms else R0[0]                                            # the same axis in the EFG's
    c2 = (1.0 + 2.0 * Atex) / 3.0
    if dn.value(c2) < 0.0:
        c2 = 0.0
    c1 = Am * np.sqrt(c2)
    r1, r2 = Ah, Ah * Ah
    lam0 = 0.5 * (3.0 * c2 - 1.0)
    lam1 = 0.5 * (2.0 * c2 + c1 - 1.0)
    lam2 = 0.25 * (1.0 + 2.0 * c1 + c2)
    AW = c1 * _outer(fl, fl) + 0.5 * (1.0 + c1) * (_I3 - _outer(fl, fl))
    AZ = _outer(n0, n0) + r1 * (_I3 - _outer(n0, n0))
    Rm = np.matmul(np.matmul(AW, R0), AZ)

    s = It[:, 0, 0] + It[:, 1, 1] + It[:, 2, 2]
    T = It - (s / 3.0)[:, None, None] * _I3
    wl = np.matmul(Gc, np.transpose(Rm))                    # <R> Gc of every line, in lab coordinates
    T1 = _rank2_scale(T, n0, 1.0, r1, r2)                   # the azimuth about n0, in the EFG's axes
    T2 = np.matmul(np.matmul(R0, T1), np.transpose(R0))     # to the lab frame
    T3 = _rank2_scale(T2, fl, lam0, lam1, lam2)             # the wobble about the lab axis
    P = dn.array([[s / 3.0 + T3[:, 0, 0], T3[:, 0, 1] + 1j * wl[:, 2]],
                  [T3[:, 0, 1] - 1j * wl[:, 2], s / 3.0 + T3[:, 1, 1]]], dtype=complex)
    return np.transpose(P, (2, 0, 1)), S, moment
