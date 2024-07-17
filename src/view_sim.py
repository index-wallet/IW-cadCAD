import sys

from PyQt6.QtWidgets import QApplication

from ui.window import MainWindow


app = QApplication(sys.argv)
window = MainWindow(
    "sim_results/7bfa8a0eb2f192a112d067518fd71f78e7cf2f57/2024-07-17 17:24:34.sim"
)
window.show()

app.exec()
