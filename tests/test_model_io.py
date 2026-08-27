"""Unit tests for the pure model-I/O helpers used by the GUI.

These functions back the parameters table and library loading, so they are pure
(no Qt needed) and worth pinning down precisely.
"""
import pytest

from syncmoss.constants import numco, number_of_baseline_parameters
from syncmoss.model_io import mod_len_def, _remap_reference_text


# Expected parameter count per model type (must match models.TImod parameter
# layout). Every anisotropic component is now the polarized model: the former
# scalar asymmetry A was replaced by orientation angles (theta_k, phi_h) + the
# uniaxial texture parameter A. 'Hamiltonian' keeps its crystal angles + alpha_k
# and carries three mosaic order parameters (A, A_m, A_h); it supersedes
# Hamilton_mc/_pc, whose counts must keep working for old files. Singlet is
# isotropic (no angles/texture); Sextet(rough)/Average_H had no polarized twin
# and keep their counts.
EXPECTED_PARAM_COUNTS = {
    "Singlet": 4,
    "Doublet": 9,
    "Sextet": 14,
    "Sextet(rough)": 14,
    "MDGD": 17,
    "Relax_2S": 14,
    "Average_H": 11,
    "Relax_MS": 11,
    "ASM": 15,
    "SCDW": 27,
    "Hamiltonian": 15,
    "Hamilton_mc": 12,       # deprecated
    "Hamilton_pc": 9,        # deprecated
    "Variables": numco,                       # 26
    "Nbaseline": number_of_baseline_parameters,  # 8
    "Layer": 0,
}


@pytest.mark.parametrize("model_name,expected", sorted(EXPECTED_PARAM_COUNTS.items()))
def test_mod_len_def_base_models(model_name, expected):
    # The "base" parameter count is independent of include_special for these models.
    assert mod_len_def(model_name, include_special=True) == expected
    assert mod_len_def(model_name, include_special=False) == expected


@pytest.mark.parametrize(
    "model_name,special_count",
    [("Distr", 5), ("Corr", 2), ("Expression", 1), ("Recon", 7)],
)
def test_mod_len_def_special_models(model_name, special_count):
    # Special models only contribute parameters when include_special is True.
    assert mod_len_def(model_name, include_special=True) == special_count
    assert mod_len_def(model_name, include_special=False) == 0


def test_mod_len_def_unknown_model_is_zero():
    assert mod_len_def("NotAModel") == 0
    assert mod_len_def("") == 0


def test_remap_reference_text_constraint():
    # new_x = z + x - number_of_baseline_parameters + 1 ; with z=5, baseline=8 -> x - 2
    assert _remap_reference_text("=[10,2]", 5) == "=[8,2]"


def test_remap_reference_text_p_reference():
    assert _remap_reference_text("p[10]", 5) == "p[8]"


def test_remap_reference_text_combined_expression():
    assert _remap_reference_text("p[10]+p[12]", 5) == "p[8]+p[10]"


def test_remap_reference_text_passthrough_for_plain_text():
    # A plain numeric literal contains no references and must be untouched.
    assert _remap_reference_text("3.14", 5) == "3.14"
    assert _remap_reference_text("", 5) == ""


def test_remap_reference_text_non_string_returns_input():
    assert _remap_reference_text(None, 5) is None
    assert _remap_reference_text(42, 5) == 42


def test_remap_reference_text_formula_is_consistent_with_constant():
    # Guard the exact remap arithmetic against accidental off-by-one changes.
    z, x = 12, 20
    expected_index = z + x - number_of_baseline_parameters + 1
    assert _remap_reference_text(f"p[{x}]", z) == f"p[{expected_index}]"
