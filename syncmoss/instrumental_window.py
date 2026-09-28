"""Standalone viewer for the instrumental function.

Opened from *Supp -> Plot instrumental function from memory / from spectrum*.
It is deliberately a separate top-level window rather than a drawing on the main
canvas: the instrumental function is not a result of the current fit, it is a
property of the source, and the user wants to keep looking at it WHILE working
on the spectrum. One window is reused for every later plot.

It carries its own matplotlib figure, canvas and navigation toolbar, so zooming
here never touches the main plot's toolbar history.
"""
import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget,
)

from syncmoss.spectrum_plotter import plot_instrumental_function


class InstrumentalFunctionWindow(QMainWindow):
    """A figure plus a short textual summary of one instrumental function."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Standalone top-level window (its own taskbar entry), not a child panel.
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle("SYNCmoss - Instrumental function")
        self.resize(1150, 640)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, 1)

        self.summary = QLabel("", self)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.summary)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.save_btn = QPushButton("Save figure", self)
        self.save_btn.clicked.connect(self._save)
        buttons.addWidget(self.save_btn)
        root.addLayout(buttons)

        self._last_saved = None

    def show_curves(self, curves, title, summary, dir_path, theme=None,
                    gridcolor='gray'):
        """Draw ``curves`` (label, v, S, color), set the summary line, raise."""
        self._last_saved = plot_instrumental_function(
            self.figure, curves, title, dir_path,
            gridcolor=gridcolor, theme=theme)
        self.canvas.draw()
        self.summary.setText(summary)
        self.setWindowTitle(f"SYNCmoss - {title}")
        self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        return self._last_saved

    def _save(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        start = self._last_saved[1] if self._last_saved else "instrumental.png"
        path, _filt = QFileDialog.getSaveFileName(
            self, "Save the instrumental-function figure", start,
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        try:
            self.figure.savefig(path, bbox_inches='tight', dpi=300,
                                facecolor=self.figure.get_facecolor())
        except Exception as e:
            QMessageBox.warning(self, "Instrumental function",
                                f"Could not save:\n{path}\n\n{e}")
