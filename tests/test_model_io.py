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
# and carries three mosaic order parameters (A, Am, Ah); it supersedes
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


# --- baseline references survive an append -----------------------------------
# An appended model keeps the DESTINATION baseline, and a baseline parameter sits
# at the same flat index in both models, so a link into the first
# number_of_baseline_parameters slots must come out byte-identical. Shifting it
# (the pre-fix behaviour) silently repointed e.g. '=[1,1]' at whatever component
# parameter happened to land on that index.

@pytest.mark.parametrize("z", [7, 21, 100])
@pytest.mark.parametrize("x", list(range(number_of_baseline_parameters)))
def test_remap_reference_text_keeps_baseline_links(x, z):
    assert _remap_reference_text(f"=[{x},1]", z) == f"=[{x},1]"
    assert _remap_reference_text(f"p[{x}]", z) == f"p[{x}]"


def test_remap_reference_text_first_component_param_is_the_boundary():
    # x == number_of_baseline_parameters is the appended model's FIRST component
    # parameter: it must shift. x one below it is the last baseline slot: it must not.
    z = 21
    first = number_of_baseline_parameters
    assert _remap_reference_text(f"p[{first}]", z) == f"p[{z + 1}]"
    assert _remap_reference_text(f"p[{first - 1}]", z) == f"p[{first - 1}]"


def test_remap_reference_text_append_after_baseline_is_identity():
    # Appending directly after the baseline (z = last baseline index) must leave
    # every reference alone, baseline or not.
    z = number_of_baseline_parameters - 1
    text = "p[0]+p[3]*p[8]-p[20]"
    assert _remap_reference_text(text, z) == text
    assert _remap_reference_text("=[8,2]", z) == "=[8,2]"


def test_remap_reference_text_mixes_baseline_and_component_links():
    z = 21
    assert (_remap_reference_text("p[0]+p[8]+p[7]+p[16]", z)
            == f"p[0]+p[{z + 1}]+p[7]+p[{z + 9}]")
