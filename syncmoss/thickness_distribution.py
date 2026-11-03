"""The 'Distr_T' row: a distribution of the sample's thickness over the beam.

Across the beam the sample is a set of patches, and the detector adds their
counts. A patch of physical thickness X scales every component below the row by
s = X/<X>, <X> being the mean of the probability density P(X) over the area: a
component's T is the mean of its local T. The patch's share of the detected beam
is P(X) e^(-X/lam), normalised: lam is the attenuation length at 14.4 keV, in
the unit of X (with 1/lam the sample's minus that of what fills the gap where
the sample is thinner), and it takes photons away from the thick patches. Nnr
is added after the transmission, so how it splits over the patches changes
nothing. docs/thickness_distribution.md has the study behind it.
"""
import numpy as np

from syncmoss.expression_eval import eval_expr


def thickness_points(text, p, lam, L, R, num):
    """The patches of a Distr_T on its grid linspace(L, R, num): their relative
    thickness s = X/<X> and their shares w of the detected beam, P(X) e^(-X/lam)
    normalised. *text* is the PDF in X; its p[i] read *p*.

    Raises ValueError when lam is 0, or when the PDF is not finite on the grid
    or has no positive sum or mean there."""
    if lam == 0:
        raise ValueError("the attenuation length λ must not be 0")
    X = np.linspace(L, R, int(num))
    P = np.asarray(eval_expr(str(text), p, X) + 0 * X, dtype=float)
    area = np.sum(P)
    if not (np.all(np.isfinite(P)) and area > 0):
        raise ValueError("the probability density function must be finite, with a positive "
                         "sum over its points from L to R")
    mean = np.sum(P * X) / area
    if not mean > 0:
        raise ValueError("the mean thickness over its points from L to R must be positive")
    w = P * np.exp(-X / lam)
    return X / mean, w / np.sum(w)
