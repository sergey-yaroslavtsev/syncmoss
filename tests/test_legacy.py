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
# The Faraday-active models (Sextet, MDGD, Relax_2S) gain Am right after A,
# so their new count is old + 3 (theta_k, phi_h, Am) rather than old + 2.
# (The two Hamiltonian components are not part of this merge -- they were merged
# with EACH OTHER into 'Hamiltonian'; see the tests at the bottom of this file.)
_MERGED = {
    'Doublet':     (7, 9, 5),
    'Sextet':      (11, 14, 6),
    'MDGD':        (14, 17, 10),
    'Relax_MS':    (9, 11, 5),
    'Relax_2S':    (11, 14, 8),
    'ASM':         (12, 15, 9),
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
    # both old Hamiltonians are now the one textured 'Hamiltonian'
    ("Hamilton_mc_(thick)", "Hamiltonian"),
    ("Hamilton_mc", "Hamiltonian"),
    ("Hamilton_pc", "Hamiltonian"),
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


# Faraday models: (pre-Am polarized count, index of A in that layout). A pre-Am
# file (theta_k/phi_h/A but no Am) must gain Am=0 right after A.
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
    assert groups[a_idx + 1][0] == '0'                # Am = 0 inserted right after A
    assert groups[a_idx + 1][1:3] == ['-1', '1']      # Am bounds [-1, 1]
    assert float(groups[a_idx + 2][0]) == a_idx + 1   # the old next param shifted by one


def test_upgrade_asm_scalar_appends_omega_at_degenerate_default():
    # Scalar (12) -> polarized: theta_k=90/phi_h=0 inserted, so the appended
    # cycloid-plane angle omega = omega_0(90, 0) = 90 (degenerate u || h fallback).
    out = legacy.upgrade_mdl_row('ASM', _row(range(12)))
    groups = [out[i * 5:i * 5 + 5] for i in range(len(out) // 5)]
    assert len(groups) == 15 == mod_len_def('ASM', include_special=False)
    assert float(groups[-1][0]) == pytest.approx(90.0)     # omega = omega_0(90, 0)
    assert groups[-1][1:3] == ['-360', '360']              # omega bounds


def test_upgrade_asm_pre_omega_polarized_appends_omega0():
    # Pre-omega polarized (14) -> new (15): omega appended = omega_0(theta_k, phi_h)
    # read from the row (here a tilted axis), everything else kept in place.
    vals = list(range(14))
    vals[9], vals[10] = 30.0, 40.0                         # theta_k, phi_h
    out = legacy.upgrade_mdl_row('ASM', _row(vals))
    groups = [out[i * 5:i * 5 + 5] for i in range(len(out) // 5)]
    assert len(groups) == 15
    assert float(groups[13][0]) == 13                      # I13 (old last param) kept
    assert float(groups[-1][0]) == pytest.approx(legacy._asm_omega0(30.0, 40.0))


def test_upgrade_asm_is_noop_for_new_layout():
    # A 15-param ASM row is already current -> returned unchanged.
    row = _row(range(15), pad_to=16)
    assert legacy.upgrade_mdl_row('ASM', row) == row


def test_upgrade_mdl_row_is_noop_for_new_layout():
    # A row already in the new layout (new count) must be returned unchanged.
    _, new_n, _ = _MERGED['Doublet']
    row = _row(range(new_n), pad_to=16)
    assert legacy.upgrade_mdl_row('Doublet', row) == row


def test_upgrade_mdl_row_is_noop_for_unmerged_model():
    row = _row(range(4), pad_to=16)
    assert legacy.upgrade_mdl_row('Singlet', row) == row


# --- the Hamiltonian merge (Hamilton_mc + Hamilton_pc -> 'Hamiltonian') --------
# A row is identified by its parameter count and padded out to 15 with the order
# parameters that reproduce the old model exactly: the powder Hamilton_pc (9)
# gains theta/phi/alpha_k = 0 and A = Am = Ah = 0, the single-crystal
# Hamilton_mc (11 before it gained alpha_k, 12 after) gains A = Am = Ah = 1.
@pytest.mark.parametrize("old_n,tail", [
    (9,  [0, 0, 0, 0, 0, 0]),      # Hamilton_pc -> random powder
    (11, [0, 1, 1, 1]),            # pre-alpha_k Hamilton_mc -> single crystal
    (12, [1, 1, 1]),               # Hamilton_mc -> single crystal
])
def test_upgrade_hamiltonian_row(old_n, tail):
    out = legacy.upgrade_mdl_row('Hamiltonian', _row(range(old_n), pad_to=16))
    groups = [out[i * 5:i * 5 + 5] for i in range(len(out) // 5)]
    assert len(groups) == 15 == mod_len_def('Hamiltonian', include_special=False)
    # the original parameters keep their places and values ...
    for i in range(old_n):
        assert float(groups[i][0]) == i
    # ... and the appended slots carry the reproducing values, all fitted-fixed.
    assert [float(g[0]) for g in groups[old_n:]] == [float(v) for v in tail]
    assert all(g[4] == 'True' for g in groups[old_n:])
    assert groups[-1][1:3] == ['0', '1']          # Ah bounds
    assert groups[-2][1:3] == ['-1', '1']         # Am bounds
    assert groups[-3][1:3] == ['-0.5', '1']       # A bounds


def test_upgrade_hamiltonian_is_noop_for_new_layout():
    row = _row(range(15), pad_to=16)
    assert legacy.upgrade_mdl_row('Hamiltonian', row) == row


def test_upgrade_mdl_row_preserves_constraint_reference_in_A():
    # A constraint/expression in the asymmetry field is left untouched (not
    # parseable as a float), while the angles are still inserted.
    old_n, _, asym = _MERGED['Doublet']
    vals = list(range(old_n))
    row = _row(vals)
    row[asym * 5] = '=[12,1]'                 # asymmetry is a reference
    out = legacy.upgrade_mdl_row('Doublet', row)
    assert out[(asym + 2) * 5] == '=[12,1]'   # kept verbatim at the new A slot
