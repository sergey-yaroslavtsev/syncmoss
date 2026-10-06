"""Parameters of the spectra: the names N, N1, N2, ... in user formulas.

In a sequence fit every spectrum may carry numbers of its own -- a temperature,
an angle, ... -- which the Expression, Distr and Corr texts use as N1, N2, ...
N is the spectrum's number in the sequence: its position in the spectrum path
box, counted from 1. While one spectrum is computed every name is replaced by
its value in the formula text ("p[9]+N1*2" -> "p[9]+(77.0)*2"), so the numbers
travel with the text to the pool workers that evaluate Distr/Corr.

Where the numbers live
----------------------
The spectrum path box. A plain path is a spectrum without parameters, a tuple
``('path', N1, N2, ...)`` one with parameters, and both may be mixed::

    [('Fe_4K.dat', 4.2, 0),
    ('Fe_77K.dat', 77, 0)]

The parameters file, loaded into and saved from the path box (Multispectra
settings menu), holds one parameter per line with one number per spectrum, separated by
spaces and/or tabs::

    #basename
    Fe_4K.dat Fe_77K.dat
    #number
    1 2
    #N1
    4.2 77
    #N2
    0 0

Lines starting with ``#`` are comments, except two reserved names:

* ``#basename`` -- the next line names the spectrum of each column (file
  basenames); the numbers are then attached by name instead of by position.
* ``#number`` -- the next line gives each spectrum's N, and the path box is
  reordered by it. It means nothing without ``#basename``.
"""
import ast
import os
import re
import shlex

import numpy as np

# N, N1, N2, ... as a whole name: not inside another name, not an attribute.
_NAME = re.compile(r'(?<![\w.])N(\d*)(?!\w)')

_RESERVED = ('basename', 'number')

# An attempt at the ('file', N1, ...) syntax: "(" before a quote, or a quote,
# a comma and a number. A bare path such as Fe (2).dat has neither.
_ENTRY_SYNTAX = re.compile(r"""\(\s*['"]|['"]\s*,\s*[-+.\d]""")

PATH_BOX_UNREADABLE = (
    "The spectrum path box could not be read. Write each spectrum as 'file', or as "
    "('file', N1, N2, …) with numbers for N1, N2, …, all inside [ ] and separated by "
    "commas -- e.g. [('a.dat', 4.2), ('b.dat', 77)]. Check that every ( has its ).")


def looks_like_entries(text):
    """True when *text* tries the ('file', N1, ...) syntax of the path box."""
    return bool(_ENTRY_SYNTAX.search(str(text)))


class SpectrumParameters:
    """What N, N1, N2, ... stand for while one spectrum is computed."""

    def __init__(self, number, values=(), path=None):
        self.number = int(number)
        self.values = tuple(float(v) for v in values)
        self.path = path

    def has(self, name):
        """True when *name* ('N', 'N1', ...) has a value for this spectrum."""
        digits = name[1:]
        if digits == '':
            return True
        return digits[0] != '0' and int(digits) <= len(self.values)


def substitute(text, parameters):
    """*text* with every N, N1, N2, ... replaced by its value, in parentheses.

    A name without a value (N3 on a spectrum with two numbers, N0) is left as
    it is, so evaluating the text reports it as an undefined name.
    """
    if parameters is None:
        return text

    def value(match):
        digits = match.group(1)
        if digits == '':
            return f'({parameters.number})'
        if parameters.has(match.group(0)):
            return f'({parameters.values[int(digits) - 1]!r})'
        return match.group(0)

    return _NAME.sub(value, str(text))


def names_used(texts):
    """Sorted names ('N', 'N1', ...) the formula *texts* use."""
    names = set()
    for text in texts:
        names.update(match.group(0) for match in _NAME.finditer(str(text)))
    return sorted(names, key=lambda name: (len(name), name))


def missing_names(texts, parameters):
    """The names *texts* use that have no value for *parameters*."""
    return [name for name in names_used(texts) if not parameters.has(name)]


# --- the spectrum path box ----------------------------------------------------

def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value))


def _literal(text):
    r"""The path box as a Python literal, with every backslash read as itself.

    Windows paths are typed with single backslashes, which Python reads as
    escapes: 'C:\Users\...' is a syntax error (\U starts a unicode escape) and
    'C:\data\new.dat' would hold a newline. In the path box a backslash is a
    backslash.
    """
    return ast.literal_eval(text.replace('\\', '\\\\'))


def _single_backslashes(path):
    r"""A path typed the Python way (C:\\data) with each doubled backslash
    counted once; the \\ that opens a network (UNC) path stays."""
    head = ''
    if path.startswith('\\\\'):
        head, path = '\\\\', path.lstrip('\\')
    return head + path.replace('\\\\', '\\')


def _entry(item):
    """One path-box item as ``(path, values)``."""
    if isinstance(item, str):
        return _single_backslashes(item), ()
    if (isinstance(item, (tuple, list)) and item and isinstance(item[0], str)
            and all(_is_number(v) for v in item[1:])):
        return _single_backslashes(item[0]), tuple(float(v) for v in item[1:])
    raise ValueError(f"{item!r} is neither a path nor a ('path', number, ...) entry")


def parse_path_box(text):
    """The path box as a list of ``(path, values)``; *values* is () for a plain path.

    Accepts a Python-style list of paths and/or ('path', N1, N2, ...) tuples, a
    single path or tuple, or plain comma-separated paths (the historical
    fallback, kept for text that is not a Python literal). Text that tries the
    tuple syntax but is not a literal (a missing bracket) raises ValueError:
    cut at its commas it would only turn into fragments reported as files.
    """
    text = text.strip()
    if not text:
        return []
    try:
        parsed = _literal(text)
    except Exception:                    # not a Python literal: a bare path, Model_<N>, ...
        if looks_like_entries(text):
            raise ValueError(PATH_BOX_UNREADABLE)
        text = text.strip("[]'\"")
        return [(_single_backslashes(p.strip().strip("'\" ")), ())
                for p in text.split(',') if p.strip()]
    if isinstance(parsed, str):
        return [_entry(parsed)]
    if isinstance(parsed, tuple) and len(parsed) > 1 and isinstance(parsed[0], str) \
            and all(_is_number(v) for v in parsed[1:]):
        return [_entry(parsed)]              # one ('path', N1, ...) on its own
    if isinstance(parsed, (list, tuple)):
        return [_entry(item) for item in parsed]
    raise ValueError(f"{parsed!r} is not a list of spectra")


def _number_text(value):
    """Shortest exact text of a float: 77.0 -> '77', 4.2 -> '4.2'."""
    text = repr(float(value))
    return text[:-2] if text.endswith('.0') else text


def _quoted(path):
    """*path* in quotes, backslashes as they are (the way the path box reads them)."""
    if "'" not in path:
        return f"'{path}'"
    if '"' not in path:
        return f'"{path}"'
    return repr(path)


def format_path_box(entries):
    """Path-box text for *entries*, one spectrum per line.

    Without any values it is exactly what Choose file writes, so a plain list
    stays ``['a.dat', 'b.dat']`` -- never ``[('a.dat',), ...]``.
    """
    items = []
    for path, values in entries:
        if values:
            items.append('(' + ', '.join([_quoted(path)] + [_number_text(v) for v in values]) + ')')
        else:
            items.append(_quoted(path))
    return '[' + ',\n'.join(items) + ']'


def path_box_entries(main_window):
    """The window's path box as ``(path, values)`` entries; [] if it cannot be read."""
    try:
        return parse_path_box(main_window.process_path.toPlainText())
    except (AttributeError, ValueError):
        return []


def first_spectrum_parameters(main_window):
    """N = 1 and the values of the first spectrum in the path box.

    The spectrum everything except a sequence works on: Show model, a single
    fit, the instrumental search, subtract-model and the pre-flight check.
    """
    entries = path_box_entries(main_window)
    if entries:
        return SpectrumParameters(1, entries[0][1], entries[0][0])
    return SpectrumParameters(1)


def table_uses_names(main_window):
    """True when an Expression/Distr/Corr text of the table uses N, N1, ..."""
    return bool(names_used(main_window.params_table.get_expression_texts().values()))


def sequence_problems(texts, entries):
    """One line per name some spectrum of a sequence has no value for."""
    lacking = {}
    for i, (path, values) in enumerate(entries):
        for name in missing_names(texts, SpectrumParameters(i + 1, values)):
            lacking.setdefault(name, []).append(os.path.basename(path))
    lines = []
    for name, files in lacking.items():
        shown = ', '.join(files[:5]) + (f' and {len(files) - 5} more' if len(files) > 5 else '')
        lines.append(f"{name} has no value for {shown}")
    return lines


# --- the parameters file ------------------------------------------------------

def _reserved_name(line):
    """'basename' / 'number' for a reserved comment line, else None."""
    word = line[1:].strip().lower()
    return word if word in _RESERVED else None


def read_parameters_file(path):
    """Read a parameters file.

    Returns ``{'basenames': [...] or None, 'numbers': [...] or None, 'rows':
    [[N1 values], [N2 values], ...]}``. Raises ValueError, naming the line, for
    anything that is not a number where a number belongs, and for ``#number``
    without ``#basename``.
    """
    with open(path, encoding='utf-8-sig') as f:
        lines = f.read().splitlines()

    table = {'basenames': None, 'numbers': None, 'rows': []}
    expecting = None                          # the reserved name whose line comes next
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            if expecting:
                raise ValueError(f"line {line_no}: the line after #{expecting} is missing")
            expecting = _reserved_name(line)
            if expecting and table[expecting + 's'] is not None:
                raise ValueError(f"line {line_no}: #{expecting} appears twice")
            continue
        if expecting == 'basename':
            try:
                table['basenames'] = shlex.split(line)
            except ValueError as e:
                raise ValueError(f"line {line_no}: the basenames could not be read ({e})")
            expecting = None
            continue
        values = []
        for token in line.split():
            try:
                value = float(token)
            except ValueError:
                raise ValueError(f"line {line_no}: '{token}' is not a number"
                                 + (" (use a decimal point, not a comma)" if ',' in token else ""))
            if not np.isfinite(value):
                raise ValueError(f"line {line_no}: '{token}' is not a finite number")
            values.append(value)
        if expecting == 'number':
            table['numbers'] = values
            expecting = None
        else:
            table['rows'].append(values)
    if expecting:
        raise ValueError(f"the line after #{expecting} is missing")
    if table['numbers'] is not None and table['basenames'] is None:
        raise ValueError("'number' is a reserved name and can be used only along with "
                         "'basename' (#basename, then a line of file basenames)")
    return table


def apply_parameters(entries, table):
    """*entries* with the values of *table*; returns ``(new_entries, notes)``.

    Without ``#basename`` the columns go to the spectra in path-box order, and
    every line must have at least one number per spectrum (extra numbers are
    ignored). With ``#basename`` each spectrum takes the column of its own
    basename, and ``#number`` then reorders the spectra; N stays the position
    in the path box, so it equals the #number when those run 1, 2, 3, ...
    Values already in the path box are replaced. *notes* are short remarks for
    the log. Raises ValueError with the reason when the table does not fit the
    spectra; nothing is changed then.
    """
    if not entries:
        raise ValueError("there are no spectra in the path box")
    n = len(entries)
    rows = table['rows']
    names = table['basenames']

    if names is None:
        short = [i + 1 for i, row in enumerate(rows) if len(row) < n]
        if short:
            raise ValueError(f"{n} spectra need {n} numbers per line, but parameter line(s) "
                             f"{', '.join(map(str, short))} have fewer")
        columns = list(range(n))
    else:
        width = len(names)
        if len(set(names)) != width:
            raise ValueError("a basename is listed twice under #basename")
        for label, row in [('#number', table['numbers'])] + [
                (f'N{i + 1}', r) for i, r in enumerate(rows)]:
            if row is not None and len(row) != width:
                raise ValueError(f"{width} basenames but {len(row)} numbers in {label}")
        own = [os.path.basename(path) for path, _ in entries]
        twice = sorted({b for b in own if own.count(b) > 1})
        if twice:
            raise ValueError(f"two spectra in the path box share the name {', '.join(twice)}: "
                             f"the file cannot tell them apart")
        missing = [b for b in own if b not in names]
        if missing:
            raise ValueError(f"not listed under #basename: {', '.join(missing)}")
        columns = [names.index(b) for b in own]

    new_entries = [(path, tuple(row[c] for row in rows))
                   for (path, _), c in zip(entries, columns)]

    notes = []
    numbers = table['numbers']
    if numbers is not None:
        mine = [numbers[c] for c in columns]
        if len(set(mine)) != len(mine):
            raise ValueError("two spectra have the same #number")
        order = sorted(range(n), key=lambda i: mine[i])
        if order != list(range(n)):
            notes.append("the spectra were reordered by #number")
        new_entries = [new_entries[i] for i in order]
        if sorted(mine) != list(range(1, n + 1)):
            notes.append(f"N counts the spectra in the path box (1 to {n}), "
                         f"not the #number values")
    return new_entries, notes


def parameters_file_text(entries, numbers=None):
    """The parameters file for *entries* (always with #basename and #number).

    *numbers* are the spectra's N; by default their position, counted from 1.
    Raises ValueError when the spectra do not all carry the same number of
    values, or when two of them share a basename.
    """
    if not entries:
        raise ValueError("there are no spectra in the path box")
    counts = {len(values) for _, values in entries}
    if len(counts) > 1:
        raise ValueError("the spectra do not all have the same number of parameters")
    names = [os.path.basename(path) for path, _ in entries]
    twice = sorted({b for b in names if names.count(b) > 1})
    if twice:
        raise ValueError(f"two spectra share the name {', '.join(twice)}")
    if numbers is None:
        numbers = range(1, len(entries) + 1)
    lines = ['#basename', '\t'.join(shlex.quote(b) for b in names),
             '#number', '\t'.join(str(int(n)) for n in numbers)]
    for k in range(counts.pop()):
        lines.append(f'#N{k + 1}')
        lines.append('\t'.join(_number_text(values[k]) for _, values in entries))
    return '\n'.join(lines) + '\n'


def write_parameters_file(path, entries, numbers=None):
    """Write :func:`parameters_file_text` to *path*."""
    text = parameters_file_text(entries, numbers)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
