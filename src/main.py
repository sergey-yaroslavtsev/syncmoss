import sys
from PyQt6.QtWidgets import QApplication
from prog_raw_qt import PhysicsApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhysicsApp()
    window.show()
    sys.exit(app.exec())