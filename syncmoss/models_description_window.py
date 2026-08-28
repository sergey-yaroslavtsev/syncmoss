"""
Markdown document viewer window (models description, quick help).

Opens a markdown document in a separate Qt window and renders it with Qt's
native markdown support so content remains selectable/copyable and follows the
active light/dark palette. The same window class serves every shipped document
(``parameters/models_description.md``, ``parameters/help.md``); only the title
and the resolved path differ.
"""

import os
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTextBrowser, QVBoxLayout, QWidget, QMainWindow


_GREEK_MAP = {
    "Alpha": "\u0391",
    "Beta": "\u0392",
    "Gamma": "\u0393",
    "Delta": "\u0394",
    "Epsilon": "\u0395",
    "Eta": "\u0397",
    "Theta": "\u0398",
    "Lambda": "\u039b",
    "Mu": "\u039c",
    "Nu": "\u039d",
    "Omega": "\u03a9",
    "Phi": "\u03a6",
    "Pi": "\u03a0",
    "Rho": "\u03a1",
    "Sigma": "\u03a3",
    "Tau": "\u03a4",
    "Chi": "\u03a7",
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "epsilon": "\u03b5",
    "varepsilon": "\u03b5",
    "eta": "\u03b7",
    "theta": "\u03b8",
    "kappa": "\u03ba",
    "lambda": "\u03bb",
    "mu": "\u03bc",
    "nu": "\u03bd",
    "omega": "\u03c9",
    "phi": "\u03c6",
    "varphi": "\u03c6",
    "pi": "\u03c0",
    "rho": "\u03c1",
    "sigma": "\u03c3",
    "tau": "\u03c4",
    "chi": "\u03c7",
}

_SUBSCRIPT_MAP = str.maketrans("0123456789+-=()", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b\u208c\u208d\u208e")
_SUPERSCRIPT_MAP = str.maketrans("0123456789+-=()", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b\u207c\u207d\u207e")

# Single-token LaTeX symbols. Applied through one alternation with a
# "not followed by a letter" guard, so no entry can be swallowed by a shorter
# one that happens to be its prefix (\in / \infty, \le / \left, \ll).
_SYMBOL_MAP = {
    "langle": "\u27e8",
    "rangle": "\u27e9",
    "times": "\u00d7",
    "cdot": "\u00b7",
    "pm": "\u00b1",
    "mp": "\u2213",
    "to": "\u2192",
    "neq": "\u2260",
    "ne": "\u2260",
    "approx": "\u2248",
    "propto": "\u221d",
    "equiv": "\u2261",
    "sim": "\u223c",
    "parallel": "\u2225",
    "perp": "\u22a5",
    "infty": "\u221e",
    "circ": "\u00b0",
    "ll": "\u226a",
    "gg": "\u226b",
    "le": "\u2264",
    "ge": "\u2265",
    "leq": "\u2264",
    "geq": "\u2265",
    "in": "\u2208",
    "rho": "\u03c1",
}
_SYMBOL_RE = re.compile(
    r"\\(" + "|".join(sorted(_SYMBOL_MAP, key=len, reverse=True)) + r")(?![A-Za-z])")


def _latex_to_plain_text(expr):
    """Convert a small, practical subset of LaTeX math to readable plain text."""
    text = (expr or "").strip()
    if not text:
        return text

    # A math span may be wrapped over several source lines; render it as one run.
    text = re.sub(r"\s+", " ", text)

    # Remove TeX spacing commands that have no plain-text meaning.
    text = re.sub(r"\\[!,;:\s]", "", text)

    # Strip common non-semantic LaTeX commands used in docs.
    text = re.sub(r"\\(?:mathbf|mathrm|mathsf|operatorname)\{([^}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")

    # Common accent/grouping commands.
    text = re.sub(r"\\hat\{([^}]*)\}", lambda m: m.group(1) + "\u0302", text)
    text = re.sub(r"\\bar\{([^}]*)\}", lambda m: m.group(1) + "\u0304", text)
    text = re.sub(r"\\sqrt\{([^}]*)\}", lambda m: "\u221a(" + m.group(1) + ")", text)
    text = re.sub(r"\\exp\s*\(", "exp(", text)

    # Degree sign: ^\circ / ^{\circ} must go before the superscript rules below,
    # which cannot translate a backslash command.
    text = re.sub(r"\^\{?\s*\\circ\s*\}?", "\u00b0", text)

    # Greek symbols.
    for name, symbol in _GREEK_MAP.items():
        text = text.replace(f"\\{name}", symbol)

    # Subscripts/superscripts.
    text = re.sub(r"_\{([^}]*)\}", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), text)
    text = re.sub(r"_([A-Za-z0-9()+\-=])", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), text)
    text = re.sub(r"\^\{([^}]*)\}", lambda m: m.group(1).translate(_SUPERSCRIPT_MAP), text)
    text = re.sub(r"\^([A-Za-z0-9()+\-=])", lambda m: m.group(1).translate(_SUPERSCRIPT_MAP), text)

    # Common symbols/operators. Matched with a trailing letter guard, so short
    # names cannot eat longer ones ('\in' must not fire inside '\infty', '\le'
    # not inside '\left').
    text = _SYMBOL_RE.sub(lambda m: _SYMBOL_MAP[m.group(1)], text)

    # Drop remaining backslashes for unknown simple commands.
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("\\", "")
    return text


# Inline math. The span may be WRAPPED over source lines (the docs hard-wrap at
# ~80 columns, and a formula that straddled a line break used to be left raw), so
# a newline is allowed inside it -- but never a blank line, which would let a
# stray unpaired '$' swallow whole paragraphs.
#
# Written as an UNROLLED loop -- one run of ordinary characters, then any number
# of (single newline + run) groups -- rather than the obvious
# ``(?:[^$\n]|\n(?!\s*\n))+?``. The obvious form nests two quantifiers over
# overlapping character sets, so an opening '$' with no reachable closer sends
# the engine into exponential backtracking; on a document of this size that is
# not a slow match but a hang. Each piece below is anchored by a mandatory '\n',
# so the match is linear.
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]*(?:\n(?!\s*\n)[^$\n]*)*)\$(?!\$)")


def _replace_latex_math_blocks(markdown_text):
    """Replace $...$ and $$...$$ with readable selectable text."""

    def repl_block(match):
        return "\n" + _latex_to_plain_text(match.group(1)) + "\n"

    def repl_inline(match):
        return _latex_to_plain_text(match.group(1))

    text = re.sub(r"\$\$([\s\S]+?)\$\$", repl_block, markdown_text)
    text = _INLINE_MATH_RE.sub(repl_inline, text)
    return text


class ModelsDescriptionWindow(QMainWindow):
    """Standalone window that displays a shipped markdown document."""

    def __init__(self, markdown_path, parent=None, title="SYNCmoss - Models description"):
        super().__init__(parent)
        self.markdown_path = markdown_path
        # Ensure this helper opens as a standalone top-level window (taskbar entry).
        self.setWindowFlag(Qt.WindowType.Window, True)

        self.setWindowTitle(title)
        self.resize(1100, 780)

        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)

        self.viewer = QTextBrowser(self)
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setReadOnly(True)
        self.viewer.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.viewer.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)

        root_layout.addWidget(self.viewer, 1)

        self.reload_document()

    def reload_document(self):
        """Load markdown from disk and render it into the viewer."""
        try:
            with open(self.markdown_path, "r", encoding="utf-8") as fh:
                markdown_text = fh.read()
        except Exception as exc:
            self.viewer.setPlainText("")
            QMessageBox.warning(self, "Models description", f"Could not load file:\n{self.markdown_path}\n\n{exc}")
            return

        # Convert LaTeX snippets into readable selectable text and keep Qt palette styling.
        self.viewer.setMarkdown(_replace_latex_math_blocks(markdown_text))


def resolve_doc_path(app_dir_path, file_name):
    """Resolve a shipped markdown document for source and frozen runs.

    The frozen bundles copy the whole ``parameters/`` folder next to the
    executable, which is the first (and normal) candidate; the rest cover a
    source checkout and the flattened layouts used while developing. The first
    candidate is returned when nothing exists, so the caller can report the
    expected location.
    """
    candidates = [
        os.path.join(app_dir_path, "parameters", file_name),
        os.path.join(app_dir_path, file_name),
        os.path.join(app_dir_path, "docs", file_name),
        os.path.join(os.path.dirname(app_dir_path), "docs", file_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def resolve_models_description_path(app_dir_path):
    """Resolve models-description markdown location for source and frozen runs."""
    return resolve_doc_path(app_dir_path, "models_description.md")


def resolve_help_path(app_dir_path):
    """Resolve the quick-help markdown location for source and frozen runs."""
    return resolve_doc_path(app_dir_path, "help.md")
