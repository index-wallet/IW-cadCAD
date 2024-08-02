import sys

from PyQt6.QtWidgets import QApplication

from ui.window import MainWindow

from util.utils import get_latest_sim

latest_sim = get_latest_sim()

app = QApplication(sys.argv)
window = MainWindow(
    latest_sim
)
window.show()

app.exec()
