"""Tests for the thin/thick merge backward-compatibility layer (syncmoss.legacy).

The scalar ("thin") models were merged into their polarized ("thick") twins: the
plain names now ARE the polarized models. ``syncmoss.legacy`` is used in exactly
one place -- opening a model file saved before the merge -- to map ``_(thick)``
names to their new names and to upgrade each pre-merge component row to the new
parameter layout (insert theta_k=90, phi_h=0; remap the scalar asymmetry A to the
texture order parameter). Everything else stores the polarized layout natively.

Pure Python/NumPy -- no Qt needed.
"""
import numpy as np
import pytest

from syncmoss import legacy
from syncmoss.model_io import mod_len_def


# (old scalar count, new count, asymmetry index in the old layout)
# The Faraday-active models (Sextet, MDGD, Relax_2S) gain A_m right after A,
# so their new count is old + 3 (theta_k, phi_h, A_m) rather than old + 2.
_MERGED = {
    'Doublet':     (7, 9, 5),
    'Sextet':      (11, 14, 6),
    'MDGD':        (14, 17, 10),
    'Relax_MS':    (9, 11, 5),
    'Relax_2S':    (11, 14, 8),
    'ASM':         (12, 14, 9),
    'Hamilton_mc': (11, 12, None),
}


def test_a_transform_anchor_points():
    # The three points the old scalar defaults map onto (see the user spec).
    assert legacy.a_scalar_to_texture(0.0) == pytest.approx(-0.25)
    assert legacy.a_scalar_to_texture(0.5) == pytest.approx(0.0)
    assert legacy.a_scalar_to_texture(1.0) == pytest.approx(1.0)


def test_a_transform_monotonic_and_bounded_on_unit_interval():
    xs = np.linspace(0.0, 1.0, 2001)
    ys = np.array([legacy.a_scalar_to_texture(x) for x in xs])
    assert np.all(np.diff(ys) > 0), "asymmetry->texture map must be strictly monotonic on [0, 1]"
    # Must stay inside the valid texture range and never dip below -0.25.
    assert ys.min() >= -0.25 - 1e-12 and ys.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("raw,expected", [
    ("Doublet_(thick)", "Doublet"),
    ("Sextet_(thick)", "Sextet"),
    ("Hamilton_mc_(thick)", "Hamilton_mc"),
    ("Doublet", "Doublet"),          # already current
    ("Singlet", "Singlet"),          # never had a thick form
    ("Doublet\r", "Doublet"),        # CR/LF stray carriage return
])
def test_normalize_legacy_model_name(raw, expected):
    assert legacy.normalize_legacy_model_name(raw) == expected


def _row(values, pad_to=None):
    """Build a flat .mdl row (groups of value, lower, upper, name, fix).

    ``pad_to`` appends empty-value groups, mimicking the table's column padding.
    """
    groups = [[str(v), "", "", "", "False"] for v in values]
    if pad_to is not None:
        groups += [["", "", "", "", "False"]] * (pad_to - len(groups))
    return [f for g in groups for f in g]


@pytest.mark.parametrize("model", sorted(_MERGED))
def test_upgrade_mdl_row_grows_to_new_layout(model):
    old_n, new_n, asym = _MERGED[model]
    row = _row(range(old_n))
    out = legacy.upgrade_mdl_row(model, row)
    assert len(out) // 5 == new_n == mod_len_def(model, include_special=False)
    if asym is not None:
        # theta_k=90 / phi_h=0 inserted at the asymmetry slot, A remapped.
        assert out[asym * 5] == '90'
        assert out[(asym + 1) * 5] == '0'
        assert float(out[(asym + 2) * 5]) == pytest.approx(legacy.a_scalar_to_texture(float(asym)))


@pytest.mark.parametrize("model", sorted(_MERGED))
def test_upgrade_mdl_row_handles_column_padding(model):
    """Real .mdl rows are padded with empty groups out to the column count; the
    upgrade must key off the leading non-empty (real) params, not the padded length.
    """
    old_n, new_n, _ = _MERGED[model]
    row = _row(range(old_n), pad_to=16)   # 16 == numco padding in real files
    out = legacy.upgrade_mdl_row(model, row)
    assert len(out) // 5 == new_n


# Faraday models: (pre-A_m polarized count, index of A in that layout). A pre-A_m
# file (theta_k/phi_h/A but no A_m) must gain A_m=0 right after A.
_PRE_AM = {'Sextet': (13, 8), 'MDGD': (16, 12), 'Relax_2S': (13, 10)}


@pytest.mark.parametrize("model", sorted(_PRE_AM))
def test_upgrade_pre_am_polarized_inserts_am_after_a(model):
    poly, a_idx = _PRE_AM[model]
    vals = list(range(poly))
    vals[a_idx] = 0.7                                  # a distinctive A value
    out = legacy.upgrade_mdl_row(model, _row(vals))
    groups = [out[i * 5:i * 5 + 5] for i in range(len(out) // 5)]
    assert len(groups) == poly + 1 == mod_len_def(model, include_special=False)
    assert float(groups[a_idx][0]) == 0.7             # A preserved
    assert groups[a_idx + 1][0] == '0'                # A_m = 0 inserted right after A
    assert groups[a_idx + 1][1:3] == ['-1', '1']      # A_m bounds [-1, 1]
    assert float(groups[a_idx + 2][0]) == a_idx + 1   # the old next param shifted by one


def test_upgrade_mdl_row_is_noop_for_new_layout():
    # A row already in the new layout (new count) must be returned unchanged.
    _, new_n, _ = _MERGED['Doublet']
    row = _row(range(new_n), pad_to=16)
    assert legacy.upgrade_mdl_row('Doublet', row) == row


def test_upgrade_mdl_row_is_noop_for_unmerged_model():
    row = _row(range(4), pad_to=16)
    assert legacy.upgrade_mdl_row('Singlet', row) == row


def test_upgrade_mdl_row_preserves_constraint_reference_in_A():
    # A constraint/expression in the asymmetry field is left untouched (not
    # parseable as a float), while the angles are still inserted.
    old_n, _, asym = _MERGED['Doublet']
    vals = list(range(old_n))
    row = _row(vals)
    row[asym * 5] = '=[12,1]'                 # asymmetry is a reference
    out = legacy.upgrade_mdl_row('Doublet', row)
    assert out[(asym + 2) * 5] == '=[12,1]'   # kept verbatim at the new A slot
