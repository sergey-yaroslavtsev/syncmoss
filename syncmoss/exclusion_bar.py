"""The exclusion-regions bar of the main window, between the plot toolbar and
the plot: the regions as text, "Pick on plot", Save and Load.

It is shown while "Apply exclusion" (the plot toolbar) is on. ``regions`` are
the applied regions; they change only through commit(), which emits
``regions_changed``. What a change does to the plot and to a fit result is the
main window's business (PhysicsApp._on_exclusion_changed).
"""
import os

import numpy as np
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QSizePolicy,
)

from syncmoss import exclusion_regions as er
from syncmoss.spectrum_plotter import (
    DATA_GID, SPECTRUM_AXES_GID, EXCLUSION_COLOR, spectrum_axes_of,
)

_FILE_FILTER = "Exclusion regions (*_exclusion.txt);;Text files (*.txt);;All files (*)"


class ExclusionBar(QWidget):
    """The text box of the exclusion regions and its buttons."""

    regions_changed = Signal()

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.regions = ()          # the applied regions
        self._first = None         # the first end of a region being picked
        self._marks = []           # its dashed lines on the plot

        # Its own height only: the plot takes every spare pixel
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        font = QFont('Arial', 14)

        label = QLabel("Exclusion regions, mm/s:")
        label.setFont(font)
        self.text = QLineEdit()
        self.text.setFont(font)
        self.text.setPlaceholderText("-3:-2; 1:2")
        self.text.setToolTip("Velocity regions whose points are not fitted: lo:hi; lo:hi\n"
                             "Enter or leaving the box applies them.")
        self.text.installEventFilter(self)
        self.text.editingFinished.connect(self.commit)
        self.text.textEdited.connect(lambda _text: self._mark_invalid(False))

        self.pick_btn = QPushButton("Pick on plot")
        self.pick_btn.setFont(font)
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Click the two ends of a region on the plot (Esc cancels)")
        self.pick_btn.toggled.connect(self._pick_toggled)
        self.clean_btn = QPushButton("Clean")
        self.clean_btn.setFont(font)
        self.clean_btn.setToolTip("Remove every exclusion region")
        self.clean_btn.clicked.connect(self.clean)
        self.save_btn = QPushButton("Save")
        self.save_btn.setFont(font)
        self.save_btn.clicked.connect(self.save)
        self.load_btn = QPushButton("Load")
        self.load_btn.setFont(font)
        self.load_btn.setToolTip("Replace the regions with those of a file")
        self.load_btn.clicked.connect(self.load)

        # The label and the buttons on one line, the regions under them
        buttons = QHBoxLayout()
        buttons.addWidget(label)
        buttons.addStretch(1)
        buttons.addWidget(self.pick_btn)
        buttons.addWidget(self.clean_btn)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.load_btn)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.text)

        # Esc cancels a pick; enabled only while picking
        self.escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), main_window)
        self.escape.setEnabled(False)
        self.escape.activated.connect(self.cancel_pick)

    # --- the text ---------------------------------------------------------------

    def eventFilter(self, obj, event):
        # Enter in the box applies the regions: it must not reach the window's
        # Return shortcut (Show model)
        if (obj is self.text and event.type() == QEvent.Type.ShortcutOverride
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def _mark_invalid(self, invalid):
        self.text.setStyleSheet("background-color: red; color: white;" if invalid else "")

    def commit(self, action_label=None):
        """Apply the text. True when it holds regions -- written back cleaned,
        and ``regions_changed`` emitted when they differ from the applied ones.
        False (and the reason in the status) when it does not, or when they
        cannot change now; then the applied regions stay. *action_label* names
        what is not started because of it."""
        try:
            regions = er.parse(self.text.text())
        except ValueError as e:
            self._mark_invalid(True)
            if action_label:
                self.main_window.set_status(
                    f"{action_label} was not started — exclusion regions: {e}", "red")
            else:
                self.main_window.set_status(f"Exclusion regions: {e}", "red")
            return False
        self._mark_invalid(False)
        if regions != self.regions and self.main_window._reject_if_busy('Changing exclusion regions'):
            self.text.setText(er.format_regions(self.regions))
            return False
        self.text.setText(er.format_regions(regions))
        if regions != self.regions:
            self.regions = regions
            self.regions_changed.emit()
        return True

    def clean(self):
        """Remove every region ("Apply exclusion" stays as it is); refused,
        like any change, while a calculation runs."""
        self.pick_btn.setChecked(False)
        self.text.setText("")
        self.commit()

    # --- Pick on plot -----------------------------------------------------------

    def _pick_toggled(self, on):
        if not on:
            self._forget_first()
            self.escape.setEnabled(False)
            return
        if self.main_window._reject_if_busy('Picking an exclusion region'):
            self.pick_btn.setChecked(False)
            return
        self.escape.setEnabled(True)
        self.main_window.set_status("Pick on plot: click the two ends of the region "
                                    "(Esc cancels)", "blue")

    def cancel_pick(self):
        if self.pick_btn.isChecked():
            self.pick_btn.setChecked(False)
            self.main_window.set_status("Picking canceled", "orange")

    def _forget_first(self):
        self._first = None
        for mark in self._marks:
            mark.remove()
        if self._marks:
            self.main_window.canvas.draw_idle()
        self._marks = []

    def on_canvas_click(self, event):
        """A click on the main plot (connected to the canvas, so it survives
        every redraw): one end of the region being picked."""
        if not self.pick_btn.isChecked() or event.button != 1:
            return
        if str(self.main_window.toolbar.mode):          # Pan or Zoom is on
            return
        ax = event.inaxes
        if ax is None or ax.get_gid() != SPECTRUM_AXES_GID or event.xdata is None:
            return
        end = self._boundary(ax, float(event.xdata))
        if self._first is None:
            self._first = end
            for spectrum_ax in spectrum_axes_of(ax.figure):
                xlim = spectrum_ax.get_xlim()
                self._marks.append(spectrum_ax.axvline(end, color=EXCLUSION_COLOR,
                                                       linestyle='--', linewidth=1))
                spectrum_ax.set_xlim(xlim)
            self.main_window.canvas.draw_idle()
            self.main_window.set_status(f"One end at {er.format_velocity(end)} mm/s: "
                                        f"click the other end (Esc cancels)", "blue")
            return
        first = self._first
        self.pick_btn.setChecked(False)                 # forgets the first end
        if first == end:
            self.main_window.set_status("Both ends are between the same two points: "
                                        "no point would be excluded", "orange")
            return
        try:
            current = er.parse(self.text.text())
        except ValueError:
            current = self.regions
        self.text.setText(er.format_regions(er.normalize(current + ((first, end),))))
        self.commit()

    def _boundary(self, ax, x):
        """Where an end clicked at *x* goes: between the data points around it,
        or, on a plot without data points, within half a pixel."""
        grid = [np.asarray(line.get_xdata(), dtype=float)
                for line in ax.lines if line.get_gid() == DATA_GID]
        if grid:
            end = er.boundary_between_points(np.concatenate(grid), x)
            if end is not None:
                return end
        x0, x1 = ax.get_xlim()
        return er.boundary_at_pixel(x, abs(x1 - x0) / max(ax.bbox.width, 1.0) / 2)

    # --- Save / Load ------------------------------------------------------------

    def save(self):
        if not self.commit():
            return
        if not self.regions:
            self.main_window.set_status("No exclusion regions to save", "orange")
            return
        base = self.main_window._result_base_path()
        default = (base + er.FILE_SUFFIX if base
                   else os.path.join(self.main_window.workfolder or "", "regions" + er.FILE_SUFFIX))
        path, _ = QFileDialog.getSaveFileName(self, "Save exclusion regions", default, _FILE_FILTER)
        if not path:
            self.main_window.set_status("Saving canceled", "orange")
            return
        try:
            er.write_file(path, self.regions)
        except OSError as e:
            self.main_window.set_status(f"Exclusion regions not saved: {e}", "red")
            return
        self.main_window.set_status(f"Exclusion regions saved to {os.path.basename(path)}", "green")

    def load(self):
        """Replace the regions with those of a file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load exclusion regions",
                                              self.main_window.workfolder or "", _FILE_FILTER)
        if not path:
            self.main_window.set_status("Loading canceled", "orange")
            return
        self.load_file(path)

    def load_file(self, path):
        try:
            regions = er.read_file(path)
        except (OSError, ValueError) as e:
            self.main_window.set_status(f"Exclusion regions not loaded from "
                                        f"{os.path.basename(path)}: {e}", "red")
            return False
        self.text.setText(er.format_regions(regions))
        if not self.commit():
            return False
        self.main_window.set_status(f"Exclusion regions loaded from {os.path.basename(path)}: "
                                    f"{er.format_regions(regions) or 'none'}", "green")
        return True
