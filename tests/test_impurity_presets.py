"""The Be / KB_nano presets and how the results table recognises them.

Be.txt and KB.txt are rounded to four decimals, like the other numbers the
parameters table shows. A model saved with the earlier presets still carries
their long digits (A = 0.427037824 where Be.txt now has 0.427) and must still be
recognised: a recognised preset is reported as the 'Impurity' and left out of
the intensity percentages of the sample's components, so missing it would
change every percentage of such a model.
"""
import os

import numpy as np
import pytest

from syncmoss import model_io

_PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "syncmoss", "parameters")


@pytest.mark.parametrize("name", ["Be.txt", "KB.txt"])
def test_presets_have_at_most_four_decimals(name):
    with open(os.path.join(_PARAMS, name), encoding="utf-8") as f:
        fields = f.read().split()
    assert len(fields) == 9                     # the polarized Doublet layout
    for field in fields:
        assert 'e' not in field.lower(), field
        assert len(field.partition('.')[2]) <= 4, field


def _percent_button(app):
    """Fill the results table the way a fit of the current table does and
    return the button that shows component 2's intensity (or 'Impurity')."""
    pt = app.params_table
    p = model_io.read_model(app)[1]
    _, fix = model_io.read_bounds_and_fix(app, len(p))
    errors = np.full(len(p), 0.01)
    errors[fix] = np.nan
    n_free = int(np.sum(~np.isnan(errors)))
    rt = app.results_table
    rt.fill_table(p, pt.get_model_list(), pt.get_current_colors(), pt.get_parameter_names(),
                  np.eye(n_free) * 1e-4, errors, fix, pt.get_expression_texts())
    return rt.buttons[2 * 3 + 1]


@pytest.mark.gui
@pytest.mark.parametrize("preset, old_a", [("Be", "0.427037824"),
                                           ("KB_nano", "-0.1166119335")])
def test_preset_is_recognised_rounded_or_with_its_old_digits(physics_app, preset, old_a):
    pt = physics_app.params_table
    pt.select_model(1, 'Sextet')
    pt.select_model(2, preset)
    assert _percent_button(physics_app).text() == 'Impurity'

    a_field = pt.row_widgets[2].layout().itemAt(7 + 1).widget().layout().itemAt(1).widget()
    a_field.setText(old_a)                      # as a model saved before the rounding has it
    assert _percent_button(physics_app).text() == 'Impurity'

    a_field.setText('0.3')                      # any other Doublet is a component
    assert _percent_button(physics_app).text() != 'Impurity'
