"""The 'Spectrum' model row: a measured reference spectrum used as a component.

The row has four slots: T, δ, G and a text. The text is the reference's line
set -- natural-width lines, each with its own T_q, δ_q, G_q, found by fitting
the reference (.dat) when the row was made -- written as one comma-separated
list T_1, δ_1, G_1, T_2, δ_2, G_2, ... Its optical depth is

    τ(E) = T · Σ_q (T_q / Σ T_q) · (π Γ0 / 2) · V(Γ0, √(G_q² + G²); E − δ_q − δ)

so T is the reference's whole thickness (Σ T_q when it is made), δ shifts every
line and G broadens every line. The flat parameters hold only a placeholder for
the text, like a Distr's PDF string: the text itself travels in read_model's
Distri list, one entry per Distr AND per Spectrum row, in model order.
"""
import numpy as np

# Rows with one entry each in read_model's Distri list, in model order: a
# Distr's PDF string, a Spectrum's line set.
DISTRI_TEXT_MODELS = ('Distr', 'Spectrum')


def distri_count(model):
    """How many Distri entries the rows of *model* (a list of names) hold."""
    return sum(1 for name in model if name in DISTRI_TEXT_MODELS)


def distr_texts(model, Distri):
    """The Distr PDF strings of a Distri list (*model*'s), without the
    Spectrum line sets it also holds."""
    kinds = [name for name in model if name in DISTRI_TEXT_MODELS]
    return [text for kind, text in zip(kinds, [] if Distri is None else Distri) if kind == 'Distr']


def spectrum_texts(model, Distri):
    """The Spectrum line sets of a Distri list (*model*'s)."""
    kinds = [name for name in model if name in DISTRI_TEXT_MODELS]
    return [text for kind, text in zip(kinds, [] if Distri is None else Distri) if kind == 'Spectrum']


def parse_spectrum_lines(text):
    """The line set of a Spectrum row's text as an (n, 3) array of T_q, δ_q,
    G_q; an empty one for an empty or unreadable text, or one whose count of
    numbers is no multiple of 3."""
    try:
        values = [float(v) for v in str(text).replace(';', ',').replace(' ', ',').split(',')
                  if v.strip()]
    except ValueError:
        return np.zeros((0, 3))
    if not values or len(values) % 3 or not np.all(np.isfinite(values)):
        return np.zeros((0, 3))
    return np.array(values, dtype=float).reshape(-1, 3)


def format_spectrum_lines(lines):
    """The text of a line set (rows T_q, δ_q, G_q), every digit kept."""
    return ','.join(repr(float(v)) for v in np.ravel(lines))
