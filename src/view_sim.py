import sys

from PyQt6.QtWidgets import QApplication

from ui.window import MainWindow


app = QApplication(sys.argv)
window = MainWindow(
    "sim_results/brute_force_powerset_search/sim_results-1719003096.8714561"
)
window.show()

app.exec()
