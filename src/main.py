import sys

from PyQt6.QtWidgets import QApplication
from sim.grid import gen_econ_network
from ui.window import MainWindow

app = QApplication(sys.argv)
window = MainWindow(gen_econ_network())
window.show()

app.exec()
