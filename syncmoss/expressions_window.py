"""
The "Load expression" window: the predefined expressions of the Expressions
folder (see expressions_io).

It lists the expressions and shows the chosen one's name, comment and text,
all three editable: Save writes them into the folder, as a new expression or
over the one of that name. A formula written in the comment between $ signs is
shown rendered (matplotlib mathtext). Every {X} of the text, and every N1, N2,
..., gets a box; what is typed there goes in when the expression is loaded, an
empty box leaves its name for the user to replace in the table. Load inserts
the result as an Expression row at the row the menu was opened on, the way the
Library inserts a model.
"""
import io
import os
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QImage, QPalette, QTextDocument
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
)

from syncmoss.expressions_io import (
    expression_name_problem, list_expressions, one_line, substitutable_names,
    substitute_expression)
from syncmoss.model_io import save_expression_model

# A formula of a comment: $...$ on one line.
_FORMULA = re.compile(r'\$([^$\n]+)\$')

# Formulas are drawn a little larger than the text around them, with this
# margin (inches) around the ink.
_FORMULA_SCALE = 1.2
_FORMULA_PAD = 0.04


def _escape(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def show_comment(view, comment):
    """Put *comment* into the QTextBrowser *view*, every ``$formula$`` drawn
    as an image in the view's text colour. A formula mathtext cannot read is
    shown as it was typed."""
    document = view.document()
    color = view.palette().color(QPalette.ColorRole.Text).name()
    ratio = view.devicePixelRatioF()
    prop = FontProperties(size=view.font().pointSizeF() * _FORMULA_SCALE)
    parts = _FORMULA.split(comment or '')
    html = []
    for k, part in enumerate(parts):
        if k % 2 == 0:                       # text between the formulas
            html.append(_escape(part).replace('\n', '<br>'))
            continue
        # Not mathtext.math_to_image: it fills the background with the figure
        # colour of the plot theme, and crops what mathtext's box leaves out
        # (an integral's lower limit). A transparent, padded tight box keeps both.
        figure = Figure()
        FigureCanvasAgg(figure)
        figure.text(0, 0, f'${part}$', fontproperties=prop, color=color)
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, dpi=view.logicalDpiY() * ratio, format='png', transparent=True,
                           bbox_inches='tight', pad_inches=_FORMULA_PAD)
        except ValueError:
            html.append(_escape(f'${part}$'))
            continue
        image = QImage.fromData(buffer.getvalue())
        url = QUrl(f'formula:{k}')
        document.addResource(QTextDocument.ResourceType.ImageResource, url, image)
        html.append(f'<img src="{url.toString()}" width="{round(image.width() / ratio)}" '
                    f'height="{round(image.height() / ratio)}" align="middle">')
    view.setHtml(''.join(html))


class ExpressionsDialog(QDialog):
    """The expressions of *main_window*.expressions_dir; Load inserts the
    chosen one at table row *insert_row*."""

    def __init__(self, main_window, insert_row):
        super().__init__(main_window)
        self.main_window = main_window
        self.insert_row = insert_row
        self.folder = main_window.expressions_dir
        self.expressions = []
        # What is typed for each name ('{X}', 'N1', ...): kept while the text
        # is edited or another expression is chosen, so nothing typed is lost.
        self.values = {}
        self.value_boxes = {}

        self.setWindowTitle("Expressions")
        self.resize(900, 640)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_widget.currentItemChanged.connect(lambda *_: self.show_expression())

        self.name_input = QLineEdit(self)
        # The comment as typed is folded away: the rendered one below is what
        # is read, the typed one only needed to change it.
        self.comment_toggle = QToolButton(self)
        self.comment_toggle.setText("Edit comment")
        self.comment_toggle.setCheckable(True)
        self.comment_toggle.setAutoRaise(True)
        self.comment_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.comment_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.comment_toggle.toggled.connect(self.show_comment_source)
        self.comment_input = QPlainTextEdit(self)
        self.comment_input.setVisible(False)
        self.comment_input.setPlaceholderText(
            "What the expression does and what each {X} stands for "
            "(recommended). Formulas between $ signs are shown rendered.")
        self.comment_input.setMaximumHeight(110)
        self.comment_view = QTextBrowser(self)
        self.comment_view.setMaximumHeight(170)
        self.expression_input = QPlainTextEdit(self)
        self.expression_input.setPlaceholderText("e.g. {a}+{b}*N1")
        self.expression_input.setMaximumHeight(80)
        self.substitution_form = QFormLayout()
        self.substitution_hint = QLabel(
            "Substitute: a box left empty keeps its name, to be replaced in the table "
            "(N1, N2, ... left as they are take each spectrum's own numbers in a sequence).", self)
        self.substitution_hint.setWordWrap(True)
        self.result_label = QLabel(self)
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.comment_input.textChanged.connect(
            lambda: show_comment(self.comment_view, self.comment_input.toPlainText()))
        self.expression_input.textChanged.connect(self.rebuild_substitutions)

        fields = QVBoxLayout()
        fields.addWidget(QLabel("Name:", self))
        fields.addWidget(self.name_input)
        fields.addWidget(self.comment_toggle)
        fields.addWidget(self.comment_input)
        fields.addWidget(QLabel("Rendered comment:", self))
        fields.addWidget(self.comment_view)
        fields.addWidget(QLabel("Expression:", self))
        fields.addWidget(self.expression_input)
        fields.addWidget(self.substitution_hint)
        fields.addLayout(self.substitution_form)
        fields.addWidget(self.result_label)
        fields.addStretch(1)

        content = QHBoxLayout()
        content.addWidget(self.list_widget, 1)
        content.addLayout(fields, 3)

        buttons = QDialogButtonBox(self)
        self.save_button = QPushButton("Save", self)
        self.load_button = QPushButton("Load", self)
        buttons.addButton(self.save_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.load_button, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        # Enter in a box loads; it never saves over a file by accident.
        self.save_button.setAutoDefault(False)
        self.load_button.setDefault(True)
        self.save_button.clicked.connect(self.save)
        self.load_button.clicked.connect(self.load)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(content)
        root.addWidget(buttons)

        self.refresh()
        self.rebuild_substitutions()

    def refresh(self, select=None):
        """Read the folder again and choose *select* (a name), else the first."""
        self.expressions, unreadable = list_expressions(self.folder)
        if unreadable:
            self.main_window.set_status(
                f"Not an expression (no Expression row): {', '.join(unreadable)}", "orange")
        self.list_widget.clear()
        for expression in self.expressions:
            self.list_widget.addItem(QListWidgetItem(expression['name']))
        # wide enough for the longest name, with room for a vertical scroll bar
        self.list_widget.setMinimumWidth(
            self.list_widget.sizeHintForColumn(0) + 2 * self.list_widget.frameWidth()
            + self.list_widget.verticalScrollBar().sizeHint().width())
        names = [expression['name'] for expression in self.expressions]
        if names:
            self.list_widget.setCurrentRow(names.index(select) if select in names else 0)

    def show_comment_source(self, shown):
        """Unfold (or fold) the comment as typed."""
        self.comment_input.setVisible(shown)
        self.comment_toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def show_expression(self):
        """Fill name, comment and text from the expression chosen in the list."""
        row = self.list_widget.currentRow()
        if not 0 <= row < len(self.expressions):
            return
        expression = self.expressions[row]
        self.name_input.setText(expression['name'])
        self.comment_input.setPlainText(expression['comment'])
        self.expression_input.setPlainText(expression['expression'])

    def expression_text(self):
        return one_line(self.expression_input.toPlainText())

    def rebuild_substitutions(self):
        """One box per name to substitute in the expression text."""
        while self.substitution_form.rowCount():
            self.substitution_form.removeRow(0)
        self.value_boxes = {}
        names = substitutable_names(self.expression_text())
        for name in names:
            box = QLineEdit(self.values.get(name, ''), self)
            box.setPlaceholderText("a number or p[i]" if name.startswith('{')
                                   else "a number, or N1/N2/... for a sequence")
            box.textChanged.connect(lambda text, n=name: self._value_changed(n, text))
            self.substitution_form.addRow(name, box)
            self.value_boxes[name] = box
        self.substitution_hint.setVisible(bool(names))
        self.update_result()

    def _value_changed(self, name, text):
        self.values[name] = text
        self.update_result()

    def result_text(self):
        """The text Load puts into the table."""
        return substitute_expression(self.expression_text(), self.values)

    def update_result(self):
        result = self.result_text()
        self.result_label.setText(f"Loads as:  {result}" if result else "")

    def save(self):
        """Write name, comment and text as <name>.mdl into the folder."""
        name = self.name_input.text().strip()
        expression = self.expression_text()
        problem = expression_name_problem(name) or (None if expression else "the expression is empty")
        if problem:
            self.main_window.set_status(f"Expression not saved: {problem}", "orange")
            return False
        path = os.path.join(self.folder, f"{name}.mdl")
        if os.path.exists(path):
            reply = QMessageBox.question(
                self, "Expressions", f"'{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                self.main_window.set_status("Saving the expression canceled", "orange")
                return False
        os.makedirs(self.folder, exist_ok=True)
        comment = self.comment_input.toPlainText().strip()
        if not save_expression_model(self.main_window, path, expression, comment or None):
            return False
        self.main_window.set_status(f"Expression '{name}' saved", "green")
        self.refresh(select=name)
        return True

    def load(self):
        """Insert the substituted text as an Expression row and close."""
        text = self.result_text()
        if not text:
            self.main_window.set_status("Nothing to load: the expression is empty", "orange")
            return
        row = self.main_window.params_table.insert_expression(self.insert_row, text)
        left = [name for name in substitutable_names(text) if name.startswith('{')]
        if left:
            self.main_window.set_status(
                f"Expression loaded into row {row}; still to substitute: {', '.join(left)}", "orange")
        else:
            self.main_window.set_status(f"Expression loaded into row {row}", "green")
        self.accept()


def open_expressions_dialog(main_window, insert_row):
    """The "Load expression" entry of a row's model menu."""
    dialog = ExpressionsDialog(main_window, insert_row)
    dialog.exec()
    dialog.deleteLater()
