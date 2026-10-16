"""Predefined expressions: the Expressions folder and the "Load expression" window.

Every expression is a .mdl with one Expression row; ``{X}`` marks a value to
substitute, and N1, N2, ... (never N) are offered for substitution too. Load
inserts the result as an Expression row, the Library's way, with the p[i] typed
by the user renumbered with the insert.
"""
import math
import os
import shutil

import numpy as np
import pytest
from scipy.integrate import quad

from syncmoss import constants
from syncmoss.expression_eval import eval_expr
from syncmoss.expressions_io import (
    expression_name_problem, list_expressions, one_line, substitutable_names,
    substitute_expression)

SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "syncmoss", "Expressions")
CENTRAL_SHIFT = "Central shift vs temperature"


def _shipped(name):
    expressions, _ = list_expressions(SHIPPED)
    return next(e for e in expressions if e['name'] == name)


# --- names to substitute ---------------------------------------------------

def test_names_are_offered_in_order_of_appearance_once_each():
    text = "{a}*N1 + {b} + N2*{a} - N1 + {N1}"
    assert substitutable_names(text) == ['{a}', 'N1', '{b}', 'N2', '{N1}']


def test_n_itself_and_names_inside_other_names_are_not_offered():
    assert substitutable_names("p[9]+N*2 + NN1 + x.N1 + N1x + p[N]") == []


def test_substitution_is_one_pass_and_keeps_empty_boxes():
    values = {'{T}': 'N1', 'N1': '300', '{a}': ''}
    # {T} -> N1 is not substituted again by N1's value; {a} stays for the table
    assert substitute_expression("{a}+{T}*N1", values) == "{a}+N1*300"


def test_a_compound_value_goes_in_parentheses():
    values = {'{T}': '-5', '{a}': 'p[3]*2', '{b}': 'p[12]', '{c}': '1.5e-3', '{d}': 'pi'}
    assert (substitute_expression("{T}**3+{a}**2+{b}+{c}+{d}", values)
            == "(-5)**3+(p[3]*2)**2+p[12]+1.5e-3+pi")


def test_a_field_holds_one_line():
    assert one_line(" a +\n\tb ") == "a + b"


def test_names_a_file_cannot_hold_are_refused():
    assert expression_name_problem("Linear in N1") is None
    assert expression_name_problem("") is not None
    assert "/" in expression_name_problem("a/b")


# --- the shipped expressions -----------------------------------------------

# Every shipped expression, with a value for each name it offers, in the order
# the window offers them.
SAMPLE_VALUES = {
    CENTRAL_SHIFT: {'{T_D}': '400', 'N1': '300', '{IS}': '0'},
    "Quadrupole splitting, doublet": {'{eps}': '0.35'},
    "Quadrupole splitting, sextet": {'{eps}': '0.1', '{eta}': '0.5', '{theta}': '30', '{phi}': '40'},
    "Paramagnetic quadrupole shift from Hamiltonian": {'{Q}': '1.2', '{eta}': '0.4'},
    "Lamb-Mossbauer factor vs temperature": {'{T_D}': '400', 'N1': '300'},
    "Lifetime from line width": {'{L}': '0.098'},
    "Hyperfine field vs temperature, critical":
        {'{H_0}': '60', 'N1': '900', '{T_C}': '950', '{beta}': '0.35'},
    "Hyperfine field vs temperature, empirical":
        {'{H_0}': '33', 'N1': '300', '{T_C}': '1043', '{alpha}': '1.5', '{beta}': '0.35'},
    "Hyperfine field vs temperature, Bloch":
        {'{H_0}': '33', '{b}': '0.1', 'N1': '300', '{T_C}': '1043'},
    "Hyperfine field in applied field, Langevin":
        {'{H_0}': '50', '{mu}': '1000', '{H_ext}': '1', 'N1': '300'},
    "Neel relaxation time vs temperature":
        {'{tau_0}': '1e-9', '{K}': '1e4', '{V}': '100', 'N1': '300'},
}


def _value(name, **changes):
    """The shipped expression *name* evaluated with SAMPLE_VALUES, ``changes``
    (``T_D=300`` for {T_D}, ``N1=4.2``) put in instead."""
    values = dict(SAMPLE_VALUES[name])
    for key, value in changes.items():
        values[key if key.startswith('N') and key[1:].isdigit() else f'{{{key}}}'] = str(value)
    return eval_expr(substitute_expression(_shipped(name)['expression'], values), np.zeros(1))


def test_every_shipped_expression_is_listed():
    expressions, unreadable = list_expressions(SHIPPED)
    assert unreadable == []
    assert sorted(e['name'] for e in expressions) == sorted(SAMPLE_VALUES)


@pytest.mark.parametrize("name", sorted(SAMPLE_VALUES))
def test_a_shipped_expression_offers_its_names_and_evaluates(name):
    expression = _shipped(name)
    assert substitutable_names(expression['expression']) == list(SAMPLE_VALUES[name])
    assert '$' in expression['comment']                   # its formula, rendered
    assert np.isfinite(_value(name))


def test_the_numbers_are_those_of_constants():
    c, k_b = constants.c, constants.K_B_EV
    assert repr(3 * c * k_b / (2 * constants.MC2_EV)) in _shipped(CENTRAL_SHIFT)['expression']
    assert (repr(6 * constants.E_RECOIL_EV / k_b)
            in _shipped("Lamb-Mossbauer factor vs temperature")['expression'])
    assert (repr(constants.HBAR_EV_S * c / constants.E0 * 1e9)
            in _shipped("Lifetime from line width")['expression'])
    assert (repr(constants.MU_B / constants.K_B)
            in _shipped("Hyperfine field in applied field, Langevin")['expression'])
    assert (repr(1e-27 / constants.K_B)
            in _shipped("Neel relaxation time vs temperature")['expression'])


def test_the_new_constants_are_the_sourced_values():
    assert constants.K_B_EV == pytest.approx(8.617333262e-5, rel=1e-10)
    assert constants.HBAR_EV_S == pytest.approx(6.582119569e-16, rel=1e-10)
    assert constants.MC2_EV == pytest.approx(53034.98265e6, rel=1e-10)
    assert constants.E_RECOIL_EV == pytest.approx(1.958e-3, rel=1e-3)   # 57Fe, E0 = 14.412 keV
    assert constants.HC_EV_A == 12398.419843320026                      # h c, eV Angstrom
    assert constants.E0_J == constants.E0 * constants.E_CHARGE


def _debye_integral(n, y):
    return quad(lambda x: x**n / np.expm1(x) if x > 0 else float(n == 1), 0, y,
                limit=200, epsabs=0, epsrel=1e-12)[0]


SOD_SLOPE = 3 * constants.c * constants.K_B_EV / (2 * constants.MC2_EV)


@pytest.mark.parametrize("T, TD", [(4.2, 400), (77, 400), (300, 400), (1000, 400), (1000, 100),
                                   (1500, 100), (3000, 50)])
def test_central_shift_follows_the_debye_model(T, TD):
    """The error of delta stays within 1e-4 of its zero-point part 3/8 slope TD
    at every temperature (the comment's figure)."""
    zero_point = -SOD_SLOPE * 3 / 8 * TD
    thermal = -SOD_SLOPE * 3 * T * (T / TD)**3 * _debye_integral(3, TD / T)
    shift = _value(CENTRAL_SHIFT, T_D=TD, N1=T)
    assert abs(shift - (zero_point + thermal)) <= 1e-4 * abs(zero_point)


def test_central_shift_at_high_temperature_is_the_classical_law():
    """T >> TD: delta = -slope (T + TD y/20 + O(y^3)), y = TD/T. The zero-point
    3/8 TD is cancelled by the y^4 term of the integral, which the approximation
    keeps exact. (Far beyond, T ~ 1000 TD, 1 - exp(-g) itself loses digits.)"""
    TD, T = 400.0, 400.0 * 50
    shift = _value(CENTRAL_SHIFT, T_D=TD, N1=T)
    assert abs(shift + SOD_SLOPE * (T + TD * (TD / T) / 20)) <= 1e-3 * SOD_SLOPE * 3 / 8 * TD


def test_central_shift_at_low_temperature_is_the_zero_point_plus_the_t4_law():
    TD, T = 400.0, 4.0
    low_T = -SOD_SLOPE * (3 / 8 * TD + 3 * T * (T / TD)**3 * math.pi**4 / 15)
    assert _value(CENTRAL_SHIFT, T_D=TD, N1=T) == pytest.approx(low_T, rel=1e-12)


def test_central_shift_adds_the_isomer_shift_at_room_temperature():
    """-0.238 mm/s of second-order Doppler shift for TD = 400 K at 300 K
    (mm/s, not eV), and the isomer shift on top."""
    assert _value(CENTRAL_SHIFT) == pytest.approx(-0.23829, abs=1e-5)
    assert _value(CENTRAL_SHIFT, IS=0.37) - _value(CENTRAL_SHIFT) == pytest.approx(0.37)


def test_doublet_quadrupole_splitting_is_twice_the_shift():
    assert _value("Quadrupole splitting, doublet", eps=-0.35) == pytest.approx(-0.7)


@pytest.mark.parametrize("eta, theta, phi", [(0.0, 0.0, 0.0), (0.0, 90.0, 0.0),
                                             (0.5, 30.0, 40.0), (0.8, 70.0, 60.0)])
def test_sextet_quadrupole_splitting_from_the_hamiltonians_first_order_shift(eta, theta, phi):
    """A weak EFG (eQV_zz = Q) under 33 T: lines 1 and 6 of the Hamiltonian move by
    +epsilon; the expression turns that back into Q/2 sqrt(1 + eta^2/3)."""
    from syncmoss import models
    Q = 1e-3
    lines = models._ham_mono_core(Q, 33.0, eta, phi, theta)[3]
    epsilon = (lines.min() + lines.max()) / 2
    splitting = _value("Quadrupole splitting, sextet", eps=epsilon, eta=eta, theta=theta, phi=phi)
    assert splitting == pytest.approx(Q / 2 * math.sqrt(1 + eta**2 / 3), rel=1e-3)


@pytest.mark.parametrize("Q, eta", [(1.2, 0.0), (1.2, 0.4), (0.8, 1.0)])
def test_paramagnetic_shift_is_where_the_hamiltonian_puts_the_lines(Q, eta):
    """At zero field the Hamiltonian's lines sit at +-epsilon (its Q is eQV_zz)."""
    from syncmoss import models
    lines = models._ham_mono_core(Q, 0.0, eta, 0.0, 0.0)[3]
    epsilon = _value("Paramagnetic quadrupole shift from Hamiltonian", Q=Q, eta=eta)
    assert epsilon == pytest.approx(lines.max(), rel=1e-9)
    assert -epsilon == pytest.approx(lines.min(), rel=1e-9)


LAMB_MOSSBAUER = "Lamb-Mossbauer factor vs temperature"


def _lm_exponent(T, TD):
    """-ln f of the shipped expression, and its scale 6E_R/(k_B TD)."""
    scale = 6 * constants.E_RECOIL_EV / constants.K_B_EV / TD
    return -np.log(_value(LAMB_MOSSBAUER, T_D=TD, N1=T)), scale


@pytest.mark.parametrize("T, TD", [(4.2, 400), (77, 400), (300, 400), (1000, 400), (1000, 100),
                                   (3000, 50), (10, 50)])
def test_lamb_mossbauer_factor_follows_the_debye_model(T, TD):
    """The error of ln f stays within 5.7e-4 of its zero-point part 3E_R/(2 k_B TD)
    at every temperature, so f is within 0.04 % for TD >= 50 K (the comment's figure)."""
    exponent, scale = _lm_exponent(T, TD)
    exact = scale * (1 / 4 + (T / TD)**2 * _debye_integral(1, TD / T))
    assert abs(exponent - exact) <= 5.7e-4 * scale / 4
    assert abs(math.expm1(exact - exponent)) <= 4e-4          # relative error of f


def test_lamb_mossbauer_factor_at_low_temperature_is_the_zero_point_plus_the_t2_law():
    TD, T = 400.0, 4.0
    exponent, scale = _lm_exponent(T, TD)
    assert exponent == pytest.approx(scale * (1 / 4 + (T / TD)**2 * math.pi**2 / 6), rel=1e-12)


def test_lamb_mossbauer_factor_at_high_temperature_is_the_classical_law():
    """T >> TD: ln f -> -6 E_R T/(k_B TD^2). The zero-point 1/4 is cancelled by
    the y^2 term of the integral, which the approximation keeps exact."""
    TD, T = 400.0, 400.0 * 1000
    exponent, scale = _lm_exponent(T, TD)
    assert abs(exponent - scale * T / TD) <= 1e-3 * scale / 4


def test_lifetime_of_the_natural_width():
    assert _value("Lifetime from line width", L=constants.NAT_WIDTH) == pytest.approx(139.71, abs=0.01)


def test_critical_field_vanishes_above_the_transition():
    name = "Hyperfine field vs temperature, critical"
    assert _value(name) == pytest.approx(60 * (50 / 950)**0.35)
    assert _value(name, N1=950) == 0.0
    assert _value(name, N1=1000) == 0.0


def test_empirical_field_with_alpha_3_2_and_beta_1_is_blochs_law():
    empirical = _value("Hyperfine field vs temperature, empirical", beta=1)
    assert empirical == pytest.approx(_value("Hyperfine field vs temperature, Bloch", b=1))
    assert _value("Hyperfine field vs temperature, empirical", N1=1100) == 0.0


def test_bloch_law_vanishes_above_the_transition():
    name = "Hyperfine field vs temperature, Bloch"
    assert _value(name) == pytest.approx(33 * (1 - 0.1 * (300 / 1043)**1.5))
    assert _value(name, N1=1100) == 0.0


LANGEVIN = "Hyperfine field in applied field, Langevin"


def test_langevin_field():
    x = 1000 * constants.MU_B * 1 / (constants.K_B * 300)
    assert _value(LANGEVIN) == pytest.approx(50 * (1 / np.tanh(x) - 1 / x), rel=1e-12)


def test_langevin_field_limits():
    """Zero applied field: no field at all (not 0/0); a weak one: H_0 x/3;
    a strong one: saturation, H_0 (1 - 1/x)."""
    assert _value(LANGEVIN, H_ext=0) == 0.0
    x = 1000 * constants.MU_B * 1e-4 / (constants.K_B * 300)
    assert _value(LANGEVIN, H_ext=1e-4) == pytest.approx(50 * x / 3, rel=1e-6)
    x = 1000 * constants.MU_B * 60 / (constants.K_B * 4.2)
    assert _value(LANGEVIN, H_ext=60, N1=4.2) == pytest.approx(50 * (1 - 1 / x), rel=1e-12)


def test_neel_relaxation_time():
    barrier = 1e4 * 100e-27 / constants.K_B                       # K V / k_B, in K
    assert (_value("Neel relaxation time vs temperature")
            == pytest.approx(1e-9 * math.exp(barrier / 300), rel=1e-12))


# --- the folder ------------------------------------------------------------

def test_a_model_without_expression_row_is_reported(tmp_path):
    shutil.copy2(os.path.join(SHIPPED, f"{CENTRAL_SHIFT}.mdl"), tmp_path)
    (tmp_path / "Hematite.mdl").write_text("baseline\tSextet\nred\tred\n1\n1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a model", encoding="utf-8")
    expressions, unreadable = list_expressions(str(tmp_path))
    assert [e['name'] for e in expressions] == [CENTRAL_SHIFT]
    assert unreadable == ["Hematite.mdl"]


def test_a_missing_folder_holds_no_expressions(tmp_path):
    assert list_expressions(str(tmp_path / "not there")) == ([], [])


def test_the_menu_offers_load_expression_after_expression():
    from syncmoss.parameters_table import MODEL_OPTIONS
    assert MODEL_OPTIONS.index('Load expression') == MODEL_OPTIONS.index('Expression') + 1


# --- the window --------------------------------------------------------------

def _window_folder(physics_app, tmp_path):
    """The shipped expression in a throw-away Expressions folder."""
    folder = tmp_path / "Expressions"
    folder.mkdir()
    shutil.copy2(os.path.join(SHIPPED, f"{CENTRAL_SHIFT}.mdl"), folder)
    physics_app.expressions_dir = str(folder)
    return str(folder)


@pytest.mark.gui
def test_the_window_reads_the_folder_next_to_the_library(physics_app):
    assert physics_app.expressions_dir == os.path.join(physics_app.dir_path, "Expressions")


@pytest.mark.gui
def test_the_typed_comment_is_folded_until_asked_for(physics_app, tmp_path):
    from syncmoss.expressions_window import ExpressionsDialog
    _window_folder(physics_app, tmp_path)
    dialog = ExpressionsDialog(physics_app, 1)
    assert dialog.comment_input.isHidden() and not dialog.comment_view.isHidden()
    assert dialog.comment_input.toPlainText()            # filled, only folded
    dialog.comment_toggle.setChecked(True)
    assert not dialog.comment_input.isHidden()
    dialog.comment_toggle.setChecked(False)
    assert dialog.comment_input.isHidden()


@pytest.mark.gui
def test_load_substitutes_and_renumbers_with_the_insert(physics_app, tmp_path):
    from syncmoss.expressions_window import ExpressionsDialog
    _window_folder(physics_app, tmp_path)
    pt = physics_app.params_table
    pt.select_model(1, 'Singlet')        # p[8] .. p[11]
    pt.select_model(2, 'Variables')      # V1 = p[12]

    dialog = ExpressionsDialog(physics_app, 2)
    assert dialog.name_input.text() == CENTRAL_SHIFT
    assert list(dialog.value_boxes) == ['{T_D}', 'N1', '{IS}']
    dialog.value_boxes['{T_D}'].setText('p[12]')       # V1, as the table numbers it now
    dialog.value_boxes['{IS}'].setText('0.37')
    template = dialog.expression_text()
    dialog.load()

    assert [pt.model_name_at(r) for r in (1, 2, 3)] == ['Singlet', 'Expression', 'Variables']
    assert sum(pt.row_params[:3]) == 13                  # V1 moved to p[13] ...
    # ... and the expression follows it; N1 is kept for a sequence
    assert pt.expression_value_input(2).text() == substitute_expression(
        template, {'{T_D}': 'p[13]', '{IS}': '0.37'})
    assert physics_app.log.toPlainText() == "Expression loaded into row 2"


@pytest.mark.gui
def test_insert_renumbers_only_what_moves(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, 'Singlet')        # p[8] .. p[11]
    pt.select_model(2, 'Singlet')        # p[12] .. p[15]
    assert pt.insert_expression(2, 'p[1]+p[9]+p[12]') == 2
    assert pt.expression_value_input(2).text() == 'p[1]+p[9]+p[13]'


@pytest.mark.gui
def test_an_unsubstituted_placeholder_is_named_in_the_log(physics_app, tmp_path):
    from syncmoss.expressions_window import ExpressionsDialog
    _window_folder(physics_app, tmp_path)
    dialog = ExpressionsDialog(physics_app, 1)
    dialog.load()
    assert "{T_D}" in physics_app.params_table.expression_value_input(1).text()
    assert physics_app.log.toPlainText() == (
        "Expression loaded into row 1; still to substitute: {T_D}, {IS}")


@pytest.mark.gui
def test_save_writes_a_model_file_the_window_and_load_model_read(physics_app, tmp_path):
    from syncmoss.expressions_window import ExpressionsDialog
    from syncmoss.model_io import load_model_from_path
    folder = _window_folder(physics_app, tmp_path)
    dialog = ExpressionsDialog(physics_app, 1)
    dialog.name_input.setText("Linear in N1")
    dialog.comment_input.setPlainText("A straight line in the temperature.\n{a} - value at 0 K")
    dialog.expression_input.setPlainText("{a} +\n{b}*N1")

    assert dialog.save()
    assert [item['name'] for item in dialog.expressions] == [CENTRAL_SHIFT, "Linear in N1"]
    assert dialog.list_widget.currentItem().text() == "Linear in N1"
    saved = next(e for e in list_expressions(folder)[0] if e['name'] == "Linear in N1")
    assert saved['comment'] == "A straight line in the temperature.\n{a} - value at 0 K"
    assert saved['expression'] == "{a} + {b}*N1"

    # an ordinary model file: "Load model" adds its Expression row
    load_model_from_path(physics_app, saved['path'], insert_row=1)
    pt = physics_app.params_table
    assert pt.model_name_at(1) == 'Expression'
    assert pt.expression_value_input(1).text() == "{a} + {b}*N1"


@pytest.mark.gui
def test_save_refuses_a_name_a_file_cannot_hold(physics_app, tmp_path):
    from syncmoss.expressions_window import ExpressionsDialog
    folder = _window_folder(physics_app, tmp_path)
    dialog = ExpressionsDialog(physics_app, 1)
    dialog.name_input.setText("a/b")
    assert dialog.save() is False
    assert physics_app.log.toPlainText().startswith("Expression not saved: a name cannot hold /")
    assert os.listdir(folder) == [f"{CENTRAL_SHIFT}.mdl"]


@pytest.mark.gui
def test_comment_formulas_are_rendered_and_a_broken_one_shown_as_typed(qapp):
    import shiboken6
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage, QTextDocument
    from PySide6.QtWidgets import QTextBrowser
    from syncmoss.expressions_window import show_comment

    view = QTextBrowser()
    try:
        show_comment(view, "shift $\\frac{x^3}{e^x-1}$ and $\\frac{$ end")
        image = view.document().resource(QTextDocument.ResourceType.ImageResource,
                                          QUrl("formula:1"))
        assert isinstance(image, QImage) and not image.isNull()
        text = view.toPlainText()
        assert "\\frac{x^3}" not in text           # drawn, not typed
        assert "$\\frac{$" in text                 # mathtext cannot read it
    finally:
        shiboken6.delete(view)
