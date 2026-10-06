"""Exclusion regions: velocity ranges whose points are left out of a fit.

A region is a pair ``(lo, hi)`` in mm/s. The regions of a fit are a tuple of
such pairs, sorted, each with ``lo < hi``, none overlapping or touching; ``()``
means none. A point is excluded when ``lo <= v <= hi``.

The model at one velocity does not depend on the other velocities computed --
the instrumental function is integrated over the source energy for every
velocity on its own (tests/test_model_pointwise.py) -- so a fit with exclusion
regions is simply a fit of the points that are left.

The user types the regions as ``lo:hi; lo:hi``; only ';' separates them.
"""
import sys

import numpy as np

FILE_SUFFIX = '_exclusion.txt'
FILE_HEADER = '# SYNCmoss exclusion regions, velocity in mm/s'


def normalize(pairs):
    """The regions of *pairs* (each in either order): sorted, overlapping or
    touching ones merged."""
    ordered = sorted((min(float(a), float(b)), max(float(a), float(b))) for a, b in pairs)
    merged = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def parse(text):
    """The regions written in *text* (``lo:hi; lo:hi``), normalized.

    Raises ValueError with a message for the user. A ',' anywhere is refused,
    so a decimal comma can never be read as something else.
    """
    text = str(text).strip()
    if ',' in text:
        raise ValueError("use ';' between regions and '.' for decimals")
    pairs = []
    for piece in text.split(';'):
        piece = piece.strip()
        if not piece:
            continue
        ends = piece.split(':')
        if len(ends) != 2:
            raise ValueError(f"'{piece}' is not a region: write it as lo:hi")
        try:
            lo, hi = float(ends[0]), float(ends[1])
        except ValueError:
            raise ValueError(f"'{piece}': both ends must be numbers") from None
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"'{piece}': both ends must be finite numbers")
        if lo == hi:
            raise ValueError(f"'{piece}' is empty: its two ends are the same")
        pairs.append((lo, hi))
    return normalize(pairs)


def format_velocity(value):
    """*value* written in the shortest form that reads back exactly."""
    return np.format_float_positional(float(value), trim='-')


def format_regions(regions):
    """``-3.05:-1.95; 0.95:2.05`` -- the text parse() reads back."""
    return '; '.join(f'{format_velocity(lo)}:{format_velocity(hi)}' for lo, hi in regions)


def kept(velocities, regions):
    """Boolean mask over *velocities*: True for the points that are fitted."""
    v = np.asarray(velocities, dtype=float)
    keep = np.ones(v.shape, dtype=bool)
    for lo, hi in regions:
        keep &= ~((v >= lo) & (v <= hi))
    return keep


def shortest_decimal_in(lo, hi, target):
    """The number with the fewest decimals strictly between *lo* and *hi*;
    of those, the one closest to *target*."""
    lo, hi = min(lo, hi), max(lo, hi)
    for decimals in range(sys.float_info.dig + 3):
        scale = 10.0 ** decimals
        first, last = int(np.floor(lo * scale)), int(np.ceil(hi * scale))
        nearest = int(min(max(round(target * scale), first), last))
        # the products above are rounded, so the candidates next to the
        # nearest one are checked as well
        for k in sorted((nearest - 1, nearest, nearest + 1),
                        key=lambda k: abs(k - target * scale)):
            if lo < k / scale < hi:
                return k / scale
    return float(target)


def boundary_between_points(grid, x):
    """Where a region boundary clicked at *x* goes: between the two points of
    *grid* around *x* (beyond an end, between the end point and one step
    further), written with the fewest decimals that tell those points apart.
    None when the grid has fewer than two points."""
    points = np.unique(np.asarray(grid, dtype=float))
    if points.size < 2:
        return None
    i = int(np.searchsorted(points, x))
    if i == 0:
        left, right = 2 * points[0] - points[1], points[0]
    elif i == points.size:
        left, right = points[-1], 2 * points[-1] - points[-2]
    else:
        left, right = points[i - 1], points[i]
    return shortest_decimal_in(left, right, x)


def boundary_at_pixel(x, half_pixel):
    """A boundary clicked at *x* on a plot without data points: the fewest
    decimals within half a pixel (*half_pixel*, in mm/s) of the click."""
    return shortest_decimal_in(x - half_pixel, x + half_pixel, x)


def write_file(path, regions):
    """Write *regions* to an exclusion-regions file (read_file reads it)."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(FILE_HEADER + '\n')
        f.write(format_regions(regions) + '\n')


def read_file(path):
    """The regions of an exclusion-regions file: its lines that are not
    comments, joined with ';'. Raises ValueError for a bad file (OSError when
    it cannot be read)."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return parse('; '.join(line for line in lines if line and not line.startswith('#')))
