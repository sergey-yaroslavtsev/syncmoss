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
  * the Faraday-active models (``Sextet``, ``MDGD``, ``Relax_2S``) additionally
    gain a magnetic polar-order parameter ``A_m = 0`` immediately after ``A``
    (unmagnetised, so the sigma+- lines keep their Faraday-averaged form, matching
    pre-A_m files);
  * ``Hamilton_mc`` had no asymmetry -- it only gains a trailing ``alpha_k = 0``;
  * ``ASM`` additionally gained a trailing cycloid-plane angle ``omega`` (both from
    the pre-merge scalar layout AND from the pre-omega polarized layout). It is
    appended set to ``omega_0(theta_k, phi_h)`` -- the angle that reproduces the
    old (plane-contains-h) geometry, up to the exact full-period symmetrization
    that the new ASM applies (see :func:`_asm_omega0` and ``models.TImod``).

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

import math


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
# ``am`` marks the Faraday-active models (Sextet, MDGD, Relax_2S) that gained a
# magnetic polar-order parameter ``A_m`` (= 0, unmagnetised) right after ``A``.
# For those, ``poly`` is the pre-A_m polarized count (theta_k/phi_h/A but no A_m)
# and ``a_idx`` the index of ``A`` in that layout, so a pre-A_m file can have A_m
# inserted in the right place too.
_MERGED = {
    'Doublet':     {'old': 7,  'asym': 5},
    'Sextet':      {'old': 11, 'asym': 6,  'am': True, 'poly': 13, 'a_idx': 8},
    'MDGD':        {'old': 14, 'asym': 10, 'am': True, 'poly': 16, 'a_idx': 12},
    'Relax_MS':    {'old': 9,  'asym': 5},
    'Relax_2S':    {'old': 11, 'asym': 8,  'am': True, 'poly': 13, 'a_idx': 10},
    # ASM gained a trailing cycloid-plane angle omega. ``trail`` marks the append;
    # ``trail_poly`` is the pre-omega polarized count and ``th_idx``/``ph_idx`` the
    # theta_k/phi_h slots in that layout, used to seed omega = omega_0(th, ph).
    'ASM':         {'old': 12, 'asym': 9, 'trail': True, 'trail_poly': 14,
                    'th_idx': 9, 'ph_idx': 10},
    'Hamilton_mc': {'old': 11, 'asym': None},
}

# Default bounds/fix for the parameters the upgrade introduces, mirroring the
# defaults ParametersTable.auto_fill_params gives a freshly selected model.
_THETA_K_BOUNDS = ('-180', '180')
_PHI_H_BOUNDS = ('-360', '360')
_A_TEX_BOUNDS = ('-0.5', '1')
_A_M_BOUNDS = ('-1', '1')
_ALPHA_K_BOUNDS = ('-360', '360')
_OMEGA_BOUNDS = ('-360', '360')


def _fmt(x):
    """Format a float back into a model-file field without noise."""
    return '%.10g' % float(x)


def _asm_omega0(theta_deg, phi_deg):
    """Cycloid-plane angle ``omega`` that reproduces the pre-omega ASM geometry.

    The old ASM built the cycloid's second in-plane axis as the transverse-to-u
    part of the radiation h = x, which corresponds to
    ``omega_0 = atan2(-sin(phi_h), cos(theta_k) * cos(phi_h))``. In the degenerate
    case u || h (theta_k = 90, phi_h = 0 or 180) the old code fell back to e = y,
    i.e. ``omega_0 = 90 deg``. See the ``ASM`` branch of ``models.TImod``.
    """
    thr = math.radians(theta_deg)
    phr = math.radians(phi_deg)
    y = -math.sin(phr)
    x = math.cos(thr) * math.cos(phr)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return 90.0
    return math.degrees(math.atan2(y, x))


def _append_trailing(model_name, info, groups):
    """Append any trailing parameter a model gained after the polarized merge.

    Currently only ``ASM``, which gained the cycloid-plane angle ``omega`` at the
    very end. ``omega`` is seeded to ``omega_0(theta_k, phi_h)`` so the upgraded
    row reproduces the old (plane-contains-h) geometry as closely as the exact
    full-period symmetrization the new ASM applies allows. A non-numeric angle
    (a constraint / expression reference) falls back to the degenerate 90 deg.
    """
    if not info.get('trail'):
        return groups
    if model_name == 'ASM':
        try:
            om = _asm_omega0(float(groups[info['th_idx']][0]),
                             float(groups[info['ph_idx']][0]))
        except (ValueError, TypeError, IndexError):
            om = 90.0
        groups = groups + [[_fmt(om), _OMEGA_BOUNDS[0], _OMEGA_BOUNDS[1], '', 'True']]
    return groups


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
    inserted where the scalar asymmetry was, that asymmetry is remapped to the
    texture order parameter, and the Faraday-active models gain ``A_m = 0`` right
    after ``A`` (Hamilton_mc instead gains a trailing ``alpha_k = 0``). If instead
    the count matches a Faraday model's PRE-A_m polarized count, only ``A_m = 0`` is
    inserted after ``A``. Otherwise (row already in the new layout, or model not
    part of the merge) it is returned unchanged, so this is safe to call
    unconditionally while loading.

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
    real_groups = groups[:real]
    asym = info['asym']
    am_group = ['0', _A_M_BOUNDS[0], _A_M_BOUNDS[1], '', 'True']

    if real == info['old']:
        # Pre-merge SCALAR row -> full polarized layout.
        if asym is None:                   # Hamilton_mc: append alpha_k (= 0)
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
            # The Faraday-active models (Sextet, MDGD, Relax_2S) gain a magnetic
            # polar-order parameter A_m = 0 immediately AFTER A (unmagnetised ->
            # Faraday-averaged sigma, i.e. the pre-A_m behaviour).
            am = [am_group] if info.get('am') else []
            upgraded = real_groups[:asym] + [theta, phi, new_a] + am + real_groups[asym + 1:]
        # ASM: append the trailing cycloid-plane angle omega (= omega_0 for the
        # theta_k/phi_h just inserted -> 90 deg for the degenerate default).
        upgraded = _append_trailing(model_name, info, upgraded)
        return [field for g in upgraded for field in g]

    if info.get('am') and real == info['poly']:
        # Pre-A_m POLARIZED row (already has theta_k/phi_h/A, but no A_m): just
        # insert A_m = 0 right after A; everything else keeps its place.
        a_idx = info['a_idx']
        upgraded = real_groups[:a_idx + 1] + [am_group] + real_groups[a_idx + 1:]
        return [field for g in upgraded for field in g]

    if info.get('trail') and real == info['trail_poly']:
        # Pre-omega POLARIZED ASM row (has theta_k/phi_h/A but no omega): append
        # omega = omega_0(theta_k, phi_h); everything else keeps its place.
        upgraded = _append_trailing(model_name, info, real_groups)
        return [field for g in upgraded for field in g]

    return row_data                        # already new layout (or unexpected)
