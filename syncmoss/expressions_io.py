"""
The predefined expressions: the Expressions folder next to parameters/ and Library/.

Every expression is a model file (.mdl) holding one Expression row, so "Load
model" and the Library's tools read it like any other model. Its name is the
file name and its comment the ``#@Comment`` lines, as for a Library model.

In the expression text ``{X}`` marks a value to substitute. The "Load
expression" window (expressions_window) offers a box for every one, and for
every N1, N2, ... -- the numbers given with each spectrum (spectrum_parameters)
-- so a temperature can be typed in for one spectrum or kept as N1 for a
sequence. N, the spectrum's number, is never offered. A box left empty leaves
its name as it is, for the user to replace in the table.
"""
import os
import re

from syncmoss.Library_io import parse_library_model_file, library_model_sort_key

# A placeholder {X}, or one of N1, N2, ... written as a whole name (the rule of
# spectrum_parameters._NAME). ONE pattern, so a substitution is one pass: a
# value put in is never substituted again, and the N1 of {N1} is the placeholder.
_SUBSTITUTABLE = re.compile(r'\{([^{}]+)\}|(?<![\w.])N\d+(?!\w)')

# A value that goes in without parentheses: a number, p[i] or a name.
_ATOM = re.compile(r'(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|p\[\d+\]|[A-Za-z_]\w*')

# Characters no file name may hold (Windows is the strictest).
FORBIDDEN_NAME_CHARACTERS = '\\/:*?"<>|'


def substitutable_names(text):
    """The names the window offers to substitute in *text*, in order of first
    appearance: ``'{X}'`` for a placeholder, ``'N1'`` for a bare N1."""
    names = []
    for match in _SUBSTITUTABLE.finditer(text or ''):
        if match.group(0) not in names:
            names.append(match.group(0))
    return names


def substitute_expression(text, values):
    """*text* with every name of :func:`substitutable_names` that has a
    non-empty value in *values* replaced by that value.

    A value that is not a single number, ``p[i]`` or name goes in parentheses,
    so ``{T}**3`` with ``-5`` for ``{T}`` reads ``(-5)**3``.
    """
    def repl(match):
        value = (values.get(match.group(0)) or '').strip()
        if not value:
            return match.group(0)
        return value if _ATOM.fullmatch(value) else f'({value})'

    return _SUBSTITUTABLE.sub(repl, text or '')


def one_line(text):
    """*text* as the one line a .mdl field can hold: a model file is
    tab-separated and line-based, so tabs and line breaks become spaces."""
    return ' '.join((text or '').split())


def expression_name_problem(name):
    """Why *name* cannot name an expression file, or None when it can."""
    if not name:
        return "it has no name"
    forbidden = [ch for ch in FORBIDDEN_NAME_CHARACTERS if ch in name]
    if forbidden:
        return f"a name cannot hold {' '.join(forbidden)}"
    return None


def read_expression_file(path):
    """``{'name', 'path', 'comment', 'expression'}`` of the expression file
    *path*, or None when the file holds no Expression row."""
    info = parse_library_model_file(path)
    expression = next((row['value'] for row in info['table_rows']
                       if row['model'] == 'Expression'), None)
    if expression is None:
        return None
    return {'name': info['title'], 'path': path,
            'comment': '\n'.join(info['comments']), 'expression': expression}


def list_expressions(folder):
    """``(expressions, unreadable)``: the expressions in *folder*, sorted by
    name like the Library, and the names of the .mdl files there that are not
    one. A folder that does not exist holds none."""
    if not os.path.isdir(folder):
        return [], []
    expressions, unreadable = [], []
    for file_name in sorted(os.listdir(folder), key=library_model_sort_key):
        path = os.path.join(folder, file_name)
        if not (os.path.isfile(path) and file_name.lower().endswith('.mdl')):
            continue
        try:
            expression = read_expression_file(path)
        except (OSError, ValueError):
            expression = None
        if expression is None:
            unreadable.append(file_name)
        else:
            expressions.append(expression)
    return expressions, unreadable
