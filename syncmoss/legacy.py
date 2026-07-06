"""Backward-compatibility helpers for the model merge (thin + polarized "thick"
-> a single polarized model per component type).

Historically each fittable component came in two flavours: a scalar ("thin")
model with a line-asymmetry parameter ``A``, and a polarized ("thick") twin in
which that asymmetry was replaced by the orientation angles ``theta_k, phi_h``
plus a uniaxial (fiber) texture order parameter ``A``. The two have now been
merged: the plain names (``Doublet``, ``Sextet``, ...) ARE the polarized model,
and the ``_(thick)`` suffix is gone.

This module exists for ONE job: opening a model file (``.mdl``) that was saved
before the merge and turning each pre-merge component row into the new polarized
layout (:func:`upgrade_mdl_row`), plus mapping any ``_(thick)`` model name to its
current plain name (:func:`normalize_legacy_model_name`). Everything else in the
codebase now stores the polarized layout natively -- the built-in ``Be`` /
``KB_nano`` presets ship in ``Be.txt`` / ``KB.txt`` as 9-value polarized doublets,
and the internal calibration / instrumental-function fits build the polarized
arrays directly -- so no runtime conversion shim is used anywhere else.

The upgrade applies the SAME transform the presets/data files were converted with:

  * insert ``theta_k = 90`` and ``phi_h = 0`` where the scalar asymmetry used to
    be, and
  * replace the scalar asymmetry ``A`` with the texture order parameter
    ``A_new = f(A)`` (see :func:`a_scalar_to_texture`);
  * ``Hamilton_mc`` had no asymmetry -- it only gains a trailing ``alpha_k = 0``.

The asymmetry->texture map ``f`` is the cubic through the three points the old
scalar defaults map onto in the new parametrisation:
``A = 0 -> -0.25``, ``A = 0.5 -> 0`` (random powder / scalar limit), ``A = 1 -> 1``
(single crystal). It is deliberately approximate -- there is no exact conversion
between a line-intensity asymmetry and a texture order parameter -- but it is
chosen to be monotonic on ``[0, 1]`` and to keep the new ``A`` inside ``[-0.25, 1]``
(so it never leaves the valid texture range ``[-0.5, 1]``).

NOTE: inserting two parameters shifts the global parameter indices of every
component after an upgraded one, so cross-component constraint references
(``=[X,Y]`` / ``p[X]``) in a pre-merge ``.mdl`` are NOT re-indexed. This is the
one acknowledged imperfection of the legacy path.
"""
from __future__ import annotations

import numpy as np


def a_scalar_to_texture(a):
    """Map an old scalar line-asymmetry ``A`` to the new texture order parameter.

    Cubic through (0, -0.25), (0.5, 0), (1, 1); monotonic on [0, 1] with range
    exactly [-0.25, 1]. See the module docstring.
    """
    return 0.5 * a ** 3 + 0.75 * a * a - 0.25


# For each model merged into its polarized twin: the OLD scalar parameter count
# and the index of the OLD scalar asymmetry ``A`` within that component (the slot
# where ``theta_k, phi_h`` are inserted and ``A`` is transformed). ``asym`` is
# None for Hamilton_mc, which had no asymmetry and only gains a trailing alpha_k.
_MERGED = {
    'Doublet':     {'old': 7,  'asym': 5},
    'Sextet':      {'old': 11, 'asym': 6},
    'MDGD':        {'old': 14, 'asym': 10},
    'Relax_MS':    {'old': 9,  'asym': 5},
    'Relax_2S':    {'old': 11, 'asym': 8},
    'ASM':         {'old': 12, 'asym': 9},
    'Hamilton_mc': {'old': 11, 'asym': None},
}

# Default bounds/fix for the parameters the upgrade introduces, mirroring the
# defaults ParametersTable.auto_fill_params gives a freshly selected model.
_THETA_K_BOUNDS = ('-180', '180')
_PHI_H_BOUNDS = ('-360', '360')
_A_TEX_BOUNDS = ('-0.5', '1')
_ALPHA_K_BOUNDS = ('-360', '360')


def _fmt(x):
    """Format a float back into a model-file field without noise."""
    return '%.10g' % float(x)


def normalize_legacy_model_name(name):
    """Map a pre-merge model name to its current name.

    The polarized models used to carry a ``_(thick)`` suffix; that suffix is gone
    now (the plain name IS the polarized model). Any other name is returned
    unchanged. A trailing ``\\r`` from CR/LF files is also stripped.
    """
    name = str(name).strip()
    if name.endswith('_(thick)'):
        name = name[:-len('_(thick)')]
    return name


def upgrade_mdl_row(model_name, row_data):
    """Upgrade one OLD-format ``.mdl`` parameter row to the new model layout.

    ``row_data`` is the flat list of string fields for one table row: groups of
    five ``[value, lower, upper, name, fix]``. Model rows in a ``.mdl`` are padded
    with empty groups out to the table's column count, so the number of *real*
    parameters is the count of leading groups with a non-empty value field.

    If the real parameter count matches the model's OLD scalar count, the row is
    upgraded to the polarized layout: ``theta_k = 90`` and ``phi_h = 0`` are
    inserted where the scalar asymmetry was, and that asymmetry is remapped to the
    texture order parameter (Hamilton_mc instead gains a trailing ``alpha_k = 0``).
    Otherwise (row already in the new layout, or model not part of the merge) it is
    returned unchanged, so this is safe to call unconditionally while loading.

    ``model_name`` must already be normalised (see
    :func:`normalize_legacy_model_name`).
    """
    info = _MERGED.get(model_name)
    if info is None:
        return row_data

    n_groups = len(row_data) // 5
    groups = [list(row_data[i * 5:i * 5 + 5]) for i in range(n_groups)]

    # Real parameters are the leading groups with a non-empty value; the rest is
    # column padding written by the table.
    real = 0
    while real < n_groups and str(groups[real][0]).strip() != '':
        real += 1
    if real != info['old']:
        return row_data                    # already new layout (or unexpected)

    real_groups = groups[:real]
    asym = info['asym']

    if asym is None:                       # Hamilton_mc: append alpha_k (= 0)
        upgraded = real_groups + [['0', _ALPHA_K_BOUNDS[0], _ALPHA_K_BOUNDS[1], '', 'True']]
    else:
        a_group = real_groups[asym]
        a_value, a_fix = a_group[0], a_group[4]
        # Best effort: only recompute a plain numeric asymmetry. A constraint /
        # expression reference (``=[..]`` / ``p[..]``) is left untouched.
        try:
            a_value = _fmt(a_scalar_to_texture(float(a_group[0])))
        except (ValueError, TypeError):
            pass
        theta = ['90', _THETA_K_BOUNDS[0], _THETA_K_BOUNDS[1], '', 'True']
        phi = ['0', _PHI_H_BOUNDS[0], _PHI_H_BOUNDS[1], '', 'True']
        new_a = [a_value, _A_TEX_BOUNDS[0], _A_TEX_BOUNDS[1], '', a_fix]
        upgraded = real_groups[:asym] + [theta, phi, new_a] + real_groups[asym + 1:]

    return [field for g in upgraded for field in g]
