import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import (
	QDialog, QWidget, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem,
	QPushButton, QLabel
)


class HamiltonianHelperWidget(QDialog):
	"""Placeholder helper window for Hamiltonian initial guess tools."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setModal(True)
		self.setWindowTitle('comming soon')
		self.resize(1100, 650)

		root_layout = QHBoxLayout(self)

		# Left side: editable 4x4 table + two buttons
		left_widget = QWidget(self)
		left_layout = QVBoxLayout(left_widget)

		title = QLabel('Hamiltonian matrix / parameters (placeholder)', left_widget)
		left_layout.addWidget(title)

		self.table = QTableWidget(4, 4, left_widget)
		for r in range(4):
			for c in range(4):
				self.table.setItem(r, c, QTableWidgetItem('0'))
		left_layout.addWidget(self.table)

		buttons_layout = QHBoxLayout()
		self.apply_btn = QPushButton('Apply', left_widget)
		self.reset_btn = QPushButton('Reset', left_widget)
		self.apply_btn.clicked.connect(self._on_apply)
		self.reset_btn.clicked.connect(self._on_reset)
		buttons_layout.addWidget(self.apply_btn)
		buttons_layout.addWidget(self.reset_btn)
		left_layout.addLayout(buttons_layout)

		# Right side: interactive matplotlib placeholder plot
		right_widget = QWidget(self)
		right_layout = QVBoxLayout(right_widget)

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, right_widget)
		right_layout.addWidget(self.toolbar)
		right_layout.addWidget(self.canvas)

		root_layout.addWidget(left_widget, 1)
		root_layout.addWidget(right_widget, 2)

		self._plot_placeholder()

	def _plot_placeholder(self):
		x = np.linspace(-15.0, 15.0, 4096)
		y = 1.0 - 0.08 * np.exp(-0.5 * (x / 1.4) ** 2)
		ax = self.figure.add_subplot(111)
		ax.clear()
		ax.plot(x, y, marker='x', linestyle='None', color='m', markersize=2)
		ax.set_xlabel('Velocity, mm/s')
		ax.set_ylabel('Transmission, counts')
		ax.grid(True, linestyle=(0, (1, 10)), linewidth=0.6)
		self.figure.tight_layout()
		self.canvas.draw_idle()

	def _on_apply(self):
		# Placeholder action for now.
		self._plot_placeholder()

	def _on_reset(self):
		for r in range(4):
			for c in range(4):
				self.table.item(r, c).setText('0')
		self._plot_placeholder()

