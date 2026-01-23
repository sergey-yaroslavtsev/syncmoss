import sys
import multiprocessing as mp
from PySide6.QtWidgets import QApplication
from prog_raw_qt import PhysicsApp

if __name__ == "__main__":
    mp.freeze_support()
    # Create global pool once at startup (reused throughout application)
    # Use same logic as original: all cores if <=4, else cores-1
    num_processes = mp.cpu_count() if mp.cpu_count() <= 4 else mp.cpu_count() - 1
    global pool
    pool = mp.Pool(processes=num_processes)
    
    app = QApplication(sys.argv)
    window = PhysicsApp(pool=pool)
    window.show()
    sys.exit(app.exec())