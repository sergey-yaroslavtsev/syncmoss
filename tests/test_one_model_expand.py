"""The simultaneous one-model fit's expansion of the template (no GUI).

The template is a model for one spectrum; one_model.expand turns it into the
Nbaseline form the simultaneous fit takes, one section per spectrum, and
OneModelResult maps the fitted expanded model back to each spectrum in the
template's layout. The template here is written by hand:

    p[0..7]   baseline: Ns free, Nnr = 0.67*Ns (=[0,0.67]), the rest fixed
    p[8..11]  Singlet A: T shared, delta INDEPENDENT =(0.1), L fixed, G shared
    p[12..15] Singlet B: T shared, delta =[9,1] (follows A's delta), L fixed,
              G =[11,2] (follows A's G)
    p[16]     Expression 'p[8]+N1*0.01+p[9]'
"""
import numpy as np
import pytest

from syncmoss import one_model
from syncmoss.constants import NBASELINE_COLOR
from syncmoss.fitting_io import recon_fit_layout
from syncmoss.model_io import fitted_model_rows, split_independent_field
from syncmoss.spectrum_parameters import SpectrumParameters

L = 17
SPECTRA = [SpectrumParameters(1, (4.2,), 'a.dat'), SpectrumParameters(2, (77.0,), 'b.dat'),
           SpectrumParameters(3, (300.0,), 'c.dat')]
BACKGROUNDS = [999.0, 2000.0, 3000.0]


def _template():
    p = np.array([1000, 0, 0, 0, 1, 0, 0, 0,          # baseline (Nnr placeholder 1)
                  1.0, 0.1, 0.097, 0.2,                # Singlet A
                  0.5, 1.0, 0.097, 1.0,                # Singlet B (links: placeholders)
                  0.0], dtype=float)                   # Expression slot
    bounds = np.array([[1] + [-np.inf] * 16, [np.inf] * 17], dtype=float)
    bounds[0][10] = bounds[0][14] = 0.097
    return {
        'model': ['Singlet', 'Singlet', 'Expression'], 'p': p,
        'con1': np.array([4, 13, 15], dtype=float), 'con2': np.array([0, 9, 11], dtype=float),
        'con3': np.array([0.67, 1.0, 2.0]),
        'Distri': [], 'Cor': [], 'Expr': ['p[8]+N1*0.01+p[9]'],
        'NExpr': np.array([16]), 'DistriN': np.array([], dtype=float),
        'Recon': [], 'ReconN': np.array([], dtype=float),
        'bounds': bounds, 'fix': np.array([1, 2, 3, 5, 6, 7, 10, 14]),
        'independent': [9],
        'colors': ['blue', 'red', 'lime', 'cyan'],
        'names': [['Ns', 'Os', 'c²s', 'lins', 'Nnr', 'Onr', 'c²nr', 'linnr'],
                  ['T', 'δ', 'L', 'G'], ['T', 'δ', 'L', 'G'], ['Expression']],
        'texts': {3: 'p[8]+N1*0.01+p[9]'},
        'links': {4: '=[0,0.67]', 9: '=(0.1)', 13: '=[9,1]', 15: '=[11,2]'},
        'rows': None,
    }


def _links(inputs):
    return sorted(zip(inputs['con1'].astype(int), inputs['con2'].astype(int), inputs['con3']))


def _apply_links_and_expressions(inputs, p):
    """What minimi_hi does with every trial vector."""
    p = np.array(p, dtype=float)
    for target, source, factor in zip(inputs['con1'], inputs['con2'], inputs['con3']):
        p[int(target)] = p[int(source)] * factor
    for text, slot in zip(inputs['Expr'], inputs['NExpr']):
        p[slot] = eval(text, {'p': p})
    return p


# --- the expansion -------------------------------------------------------------

def test_shared_slots_are_the_free_ones_outside_the_baseline():
    assert one_model.shared_slots(_template()) == [8, 11, 12]


def test_one_section_per_spectrum_each_as_long_as_the_template():
    inputs = one_model.expand(_template(), SPECTRA, BACKGROUNDS)
    assert inputs['model'] == ['Singlet', 'Singlet', 'Expression',
                               'Nbaseline', 'Singlet', 'Singlet', 'Expression',
                               'Nbaseline', 'Singlet', 'Singlet', 'Expression']
    assert len(inputs['p']) == 3 * L and inputs['L'] == L
    assert inputs['bounds'].shape == (2, 3 * L)
    assert np.array_equal(inputs['bounds'][:, L:2 * L], _template()['bounds'])


def test_the_first_spectrum_keeps_the_table_and_the_others_their_own_counts():
    p = one_model.expand(_template(), SPECTRA, BACKGROUNDS)['p']
    assert np.array_equal(p[:L], _template()['p'])
    assert (p[L], p[2 * L]) == (2000.0, 3000.0)
    assert np.array_equal(p[L + 1:2 * L], _template()['p'][1:])


def test_links_stay_inside_their_spectrum_unless_they_point_at_a_shared_parameter():
    links = _links(one_model.expand(_template(), SPECTRA, BACKGROUNDS))
    section_2 = [link for link in links if L <= link[0] < 2 * L]
    assert section_2 == [
        (L + 4, L + 0, 0.67),      # Nnr follows THIS spectrum's Ns
        (L + 8, 8, 1.0),           # shared T of A: follows spectrum 1
        (L + 11, 11, 1.0),         # shared G of A
        (L + 12, 12, 1.0),         # shared T of B
        (L + 13, L + 9, 1.0),      # B's delta follows THIS spectrum's independent delta
        (L + 15, 11, 2.0),         # B's G follows the shared G of A: spectrum 1's slot
    ]
    # no link follows another link (minimi_hi applies them in one pass)
    targets = {link[0] for link in links}
    assert not any(source in targets for _, source, _ in links)


def test_independent_and_fixed_parameters_are_copied():
    inputs = one_model.expand(_template(), SPECTRA, BACKGROUNDS)
    fix = set(inputs['fix'].tolist())
    assert {L + 10, L + 14, 2 * L + 10} <= fix                # fixed L's of every copy
    assert L + 9 not in fix and L + 9 not in inputs['con1']   # each spectrum's delta is free
    assert inputs['p'][L + 9] == 0.1                           # ... starting from X


def test_formulas_take_each_spectrum_own_values_and_renumbered_slots():
    inputs = one_model.expand(_template(), SPECTRA, BACKGROUNDS)
    assert inputs['Expr'] == ['p[8]+(4.2)*0.01+p[9]',
                              f'p[8]+(77.0)*0.01+p[{L + 9}]',
                              f'p[8]+(300.0)*0.01+p[{2 * L + 9}]']
    assert inputs['NExpr'].tolist() == [16, L + 16, 2 * L + 16]


def test_after_the_links_every_copy_of_a_shared_parameter_is_spectrum_1s():
    inputs = one_model.expand(_template(), SPECTRA, BACKGROUNDS)
    p = inputs['p'].copy()
    p[8], p[11], p[12] = 1.7, 0.3, 0.9                        # shared, as fitted
    p[9], p[L + 9], p[2 * L + 9] = 0.11, 0.22, 0.33           # independent, as fitted
    p = _apply_links_and_expressions(inputs, p)
    for k in (1, 2):
        assert p[k * L + 8] == 1.7 and p[k * L + 11] == 0.3 and p[k * L + 12] == 0.9
        assert p[k * L + 15] == 0.6                           # 2 x the shared G
    assert [p[k * L + 13] for k in range(3)] == [0.11, 0.22, 0.33]
    assert p[L + 16] == pytest.approx(1.7 + 0.77 + 0.22)
    assert p[L + 4] == pytest.approx(0.67 * 2000.0)


def test_a_recon_shares_the_first_spectrum_s_weights():
    template = _template()
    template['model'] = ['Singlet', 'Recon']
    template['p'] = np.concatenate([template['p'][:12], [1, -1, 1, 5, 0.3, 0.2, 0]])
    template.update(con1=np.array([4.]), con2=np.array([0.]), con3=np.array([0.67]),
                    Expr=[], NExpr=np.array([], dtype=int), ReconN=np.array([18.]),
                    Recon=[np.full(5, 0.2)], independent=[], fix=np.array([1, 2, 3, 5, 6, 7]),
                    bounds=np.array([[-np.inf] * 19, [np.inf] * 19]))
    inputs = one_model.expand(template, SPECTRA, BACKGROUNDS)
    assert inputs['recon_share'] == [0, 0, 0]
    infos, n_weights = recon_fit_layout(inputs['model'], inputs['p'], share=inputs['recon_share'])
    assert n_weights == 5 and len(infos) == 1
    assert infos[0]['followers'] == [1, 2]


def test_recon_layout_without_sharing_is_as_before():
    model = ['Recon', 'Nbaseline', 'Recon']
    p = np.zeros(8 + 7 + 8 + 7)
    p[8 + 3] = p[8 + 7 + 8 + 3] = 4
    infos, n_weights = recon_fit_layout(model, p)
    assert n_weights == 8 and [info['wstart'] for info in infos] == [0, 4]
    assert all(info['followers'] == [] for info in infos)


# --- the result, spectrum by spectrum ------------------------------------------

def _result():
    """A fake fit of the expansion: the free parameters moved, errors 0.01*(i+1)."""
    template = _template()
    inputs = one_model.expand(template, SPECTRA, BACKGROUNDS)
    p = inputs['p'].copy()
    p[8], p[11], p[12] = 1.7, 0.3, 0.9
    p[9], p[L + 9], p[2 * L + 9] = 0.11, 0.22, 0.33
    p = _apply_links_and_expressions(inputs, p)
    fixed = set(inputs['fix'].tolist()) | set(inputs['con1'].astype(int).tolist()) \
        | set(inputs['NExpr'].tolist())
    free = [i for i in range(len(p)) if i not in fixed]
    errors = np.full(len(p), np.nan)
    errors[free] = 0.01 * (np.array(free) + 1)
    rng = np.random.default_rng(1)
    root = rng.normal(size=(len(free), len(free)))
    result = {'parameters': p, 'errors': errors, 'covariance_matrix': root @ root.T,
              'chi2': 1.1, 'chi2_spread': 0.05, 'model': inputs['model'],
              'Distri_substituted': [], 'Cor_substituted': [], 'Recon': []}
    return one_model.OneModelResult(template, SPECTRA, inputs, result), free


def test_each_spectrum_is_its_section_of_the_result():
    result, _ = _result()
    values = result.values(1)
    assert len(values) == L
    assert values[0] == 2000.0 and values[9] == 0.22 and values[8] == 1.7
    assert values[13] == 0.22                                  # its own linked delta


def test_a_shared_parameter_has_spectrum_1s_error_in_every_spectrum():
    result, _ = _result()
    errors = result.errors_of(2)
    assert errors[8] == pytest.approx(0.01 * 9)                # slot 8 of spectrum 1
    assert errors[9] == pytest.approx(0.01 * (2 * L + 9 + 1))  # its own delta
    assert errors[0] == pytest.approx(0.01 * (2 * L + 1))      # its own Ns
    assert np.isnan(errors[10]) and np.isnan(errors[13])       # fixed, linked


def test_the_covariance_of_a_spectrum_is_the_matching_block():
    result, free = _result()
    k = 1
    view = [i for i in range(L) if not np.isnan(result.errors_of(k)[i])]
    expanded = [i if i in result.shared else k * L + i for i in view]
    rows = [free.index(i) for i in expanded]
    assert np.array_equal(result.covariance_of(k), result.covariance[np.ix_(rows, rows)])


def test_the_template_view_is_spectrum_1_with_the_snapshot_links():
    result, _ = _result()
    view = result.template_view()
    assert view['model_list'] == ['baseline', 'Singlet', 'Singlet', 'Expression']
    assert np.array_equal(view['parameters'], result.values(0))
    assert view['links'][9] == '=(0.1)'                        # X stays the start value
    assert view['texts'][3] == 'p[8]+N1*0.01+p[9]'             # as typed


def test_a_spectrum_s_texts_are_those_of_the_whole_model():
    """Each spectrum's formula as the fitted model has it (its own delta's slot,
    the shared one, its N1), evaluated with the whole fit."""
    result, _ = _result()
    assert result.texts_of(0) == {3: 'p[8]+(4.2)*0.01+p[9]'}
    assert result.texts_of(1) == {3: f'p[8]+(77.0)*0.01+p[{L + 9}]'}
    assert result.texts_of(2) == {3: f'p[8]+(300.0)*0.01+p[{2 * L + 9}]'}
    p, errors, covariance = result.whole_fit()
    assert p is result.p and len(errors) == len(p) == 3 * L


def test_the_expanded_lists_follow_the_sections():
    result, _ = _result()
    models = result.expanded_model_list()
    assert len(result.expanded_colors()) == len(result.expanded_names()) == len(models)
    assert models[4] == 'Nbaseline' and result.expanded_colors()[4] == NBASELINE_COLOR
    assert result.section_of_component(5) == (1, 1)            # spectrum 2's Singlet A
    assert result.section_of_component(4) == (1, 0)            # spectrum 2's baseline
    assert result.expanded_texts()[7] == f'p[8]+(77.0)*0.01+p[{L + 9}]'
    assert result.displayed() == [0, 2]
    assert result.parameters_line(1) == "N = 2   N1 = 77"


# --- the independent field ------------------------------------------------------

@pytest.mark.parametrize("text, value", [
    ('=(0.5)', 0.5), ('=(-3)', -3.0), (' =(12.25) ', 12.25),
    ('=()', None), ('=(', None), ('=(1', None), ('0.5', None), ('=[3,1]', None),
])
def test_split_independent_field(text, value):
    assert split_independent_field(text) == value


def test_fitted_rows_keep_or_refit_an_independent_value():
    rows = (['baseline', 'Singlet'], ['', 'blue'],
            [[['1000', '', '', '', 'False']] + [['0', '', '', '', 'True']] * 7,
             [['1', '0', '', '', 'False'], ['=(0.1)', '', '', '', 'False'],
              ['0.097', '', '', '', 'True'], ['0.2', '0', '', '', 'False']]])
    fitted = np.arange(12, dtype=float) + 0.5
    single = fitted_model_rows(rows, fitted)
    assert single[2][1][1][0] == '=(9.5)'                      # one spectrum: refitted X
    kept = fitted_model_rows(rows, fitted, keep_independent=True)
    assert kept[2][1][1][0] == '=(0.1)'                        # one-model: the start value
    assert kept[2][1][0][0] == '8.5'                           # the rest as always
