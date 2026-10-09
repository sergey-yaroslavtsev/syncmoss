"""Chains of links =[X,Y], and the order of the Expressions.

A link may point at another link: C = [B, f2] with B = [A, f1] is C = f1*f2*A.
Everything that applies the links does it in ONE pass (minimi_hi on every
trial point and in its Jacobian, the fit's start, Show model), so read_model
hands every chain on as a link to its end (model_io.resolve_link_chains). A
chain that comes back round never ends: Show model / Fit refuse it.

The Expressions are calculated one by one from the top of the table, so one
that uses an Expression below it -- or itself -- would take a stale value:
Show model / Fit refuse it and ask for the calculation order.
"""
import numpy as np
import pytest

import syncmoss.minimi_lib as mi
from syncmoss.model_io import read_model, resolve_link_chains

# Flat indices: the baseline p[0..7], a Singlet in row 1 (p[8..11]), below it a
# Doublet (p[12..20]) or two Expressions (one slot each, p[12] and p[13])
SINGLET_DELTA = 9


# --- resolve_link_chains -----------------------------------------------------

def test_links_without_a_chain_are_untouched():
    con2, con3, looped = resolve_link_chains([13.0, 14.0], [9.0, 10.0], [0.5, 3.0])
    assert con2.tolist() == [9.0, 10.0]
    assert con3.tolist() == [0.5, 3.0]
    assert looped == {}


def test_a_chain_is_handed_on_to_its_end_with_the_factors_multiplied():
    # 13 -> 14 -> 15 -> 9: each link sits BEFORE the one it follows
    con2, con3, looped = resolve_link_chains([13, 14, 15], [14, 15, 9], [2.0, 0.5, 3.0])
    assert con2.tolist() == [9.0, 9.0, 9.0]
    assert con3.tolist() == [3.0, 1.5, 3.0]
    assert looped == {}


@pytest.mark.parametrize("con1, con2, paths", [
    ([13], [13], {13: [13, 13]}),                                   # A -> A
    ([13, 14], [14, 13], {13: [13, 14, 13], 14: [14, 13, 14]}),     # A -> B -> A
    ([12, 13, 14], [13, 14, 13],                                    # A -> B -> C -> B
     {12: [12, 13, 14, 13], 13: [13, 14, 13], 14: [14, 13, 14]}),
])
def test_a_chain_that_comes_back_round_is_a_loop(con1, con2, paths):
    con2_out, _con3, looped = resolve_link_chains(con1, con2, [1.0] * len(con1))
    assert looped == paths
    assert con2_out.tolist() == [float(s) for s in con2]           # left as they were


def test_a_fit_follows_a_chain_written_in_reverse_order():
    """p[1] = 1*p[2] and p[2] = 2*p[0]. minimi_hi applies links in one pass: it
    took p[1] from the previous p[2], and its Jacobian never carried a step of
    p[0] on to p[1] -- the one the model uses -- so p[0] could not move.
    Resolved, p[1] = 2*p[0], and the fit finds p[0]."""
    x = np.linspace(0.0, 1.0, 20)
    y = 3.0 * x + 1.0

    def model(_x, p):
        return p[1] * _x + p[3]

    con1 = [1, 2]
    con2, con3, _looped = resolve_link_chains(con1, [2, 0], [1.0, 2.0])
    p, _er, _chi2, _cov = mi.minimi_hi(model, x, y, np.array([1.0, 2.0, 2.0, 0.5]),
                                      fix=np.array(con1), confu=np.array([con1, con2, con3]))
    assert p[0] == pytest.approx(1.5, rel=1e-6)
    assert p[1] == pytest.approx(3.0, rel=1e-6)
    assert p[2] == pytest.approx(3.0, rel=1e-6)


# --- the table -----------------------------------------------------------------

def _log(app):
    return app.log.toPlainText()


@pytest.mark.gui
def test_read_model_hands_a_chain_on_and_the_fit_may_start(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Doublet")
    pt._value_input(2, 1).setText("=[14,2]")                       # p[13] = 2 p[14] ...
    pt._value_input(2, 2).setText(f"=[{SINGLET_DELTA},0.5]")      # ... p[14] = 0.5 p[9]

    assert pt.get_link_loops() == []
    assert physics_app.check_user_expressions("Fit") is True
    _model, _p, con1, con2, con3, *_rest = read_model(physics_app)
    # the baseline's default Onr/c²nr/linnr -> Os/c²s/lins links come first
    assert con1.tolist() == [5.0, 6.0, 7.0, 13.0, 14.0]
    assert con2.tolist() == [1.0, 2.0, 3.0, 9.0, 9.0]
    assert con3.tolist() == [1.0, 1.0, 1.0, 1.0, 0.5]


@pytest.mark.gui
def test_links_round_a_loop_are_refused_and_marked_red(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt.select_model(2, "Doublet")
    pt._value_input(2, 0).setText("=[13,1]")      # p[12] leads into ...
    pt._value_input(2, 1).setText("=[14,1]")      # ... the loop p[13] -> p[14] -> p[13]
    pt._value_input(2, 2).setText("=[13,1]")

    assert [loop['path'] for loop in pt.get_link_loops()] == [
        [12, 13, 14, 13], [13, 14, 13], [14, 13, 14]]
    assert physics_app.check_user_expressions("Show model") is False
    assert all("red" in pt._value_input(2, col).styleSheet() for col in (0, 1, 2))
    assert "round a loop: 12 → 13 → 14 → 13" in _log(physics_app)

    pt._value_input(2, 2).setText("0.5")           # a number ends the chain
    assert physics_app.check_user_expressions("Show model") is True


@pytest.mark.gui
def test_a_link_to_itself_is_a_loop(physics_app):
    pt = physics_app.params_table
    pt.select_model(1, "Singlet")
    pt._value_input(1, 1).setText(f"=[{SINGLET_DELTA},1]")

    assert physics_app.check_user_expressions("Fit") is False
    assert f"round a loop: {SINGLET_DELTA} → {SINGLET_DELTA}" in _log(physics_app)


def _expressions(app, *texts):
    """A Singlet in row 1 and an Expression per text below it. The texts are
    typed once every row is there: adding a row renumbers the p[i] behind it."""
    pt = app.params_table
    pt.select_model(1, "Singlet")
    for row in range(2, len(texts) + 2):
        pt.select_model(row, "Expression")
    for row, text in enumerate(texts, start=2):
        pt.expression_value_input(row).setText(text)


@pytest.mark.gui
def test_an_expression_using_one_below_it_is_refused(physics_app):
    _expressions(physics_app, "p[13] + 1", "2 * p[9]")     # row 2 uses row 3's slot

    assert physics_app.check_user_expressions("Fit") is False
    log = _log(physics_app)
    assert ("Expression (table row 2) uses the Expression of table row 3, which is "
            "calculated after it") in log
    assert "put them in calculation order" in log
    assert "red" in physics_app.params_table.expression_value_input(2).styleSheet()
    assert "red" not in physics_app.params_table.expression_value_input(3).styleSheet()

    # The same two, in calculation order
    physics_app.params_table.expression_value_input(2).setText("2 * p[9]")
    physics_app.params_table.expression_value_input(3).setText("p[12] + 1")
    assert physics_app.check_user_expressions("Fit") is True


@pytest.mark.gui
@pytest.mark.parametrize("link, text, used", [
    (None, "p[12] + 1", "uses its own value"),
    ("=[12,1]", "p[10] + 1", "uses its own value"),                          # through a link
    ("=[13,1]", "p[10] * 2", "uses the Expression of table row 3"),          # through a link
])
def test_an_expression_using_itself_or_one_below_through_a_link_is_refused(physics_app, link, text, used):
    _expressions(physics_app, text, "2 * p[9]")
    if link is not None:
        physics_app.params_table._value_input(1, 2).setText(link)     # the Singlet's p[10]

    assert physics_app.check_user_expressions("Show model") is False
    assert f"Expression (table row 2) {used}" in _log(physics_app)
