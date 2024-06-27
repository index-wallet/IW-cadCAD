import sys

from PyQt6.QtWidgets import QApplication

from ui.window import MainWindow


app = QApplication(sys.argv)
window = MainWindow(
    "sim_results/814cc652373295dc0accae4aafa62275fadd7362/2024-06-26 15:38:59.sim"
)
window.show()

app.exec()
