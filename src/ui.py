from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QGridLayout,
    QMainWindow,
    QWidget,
)


class Color(QWidget):

    def __init__(self, red: int, green: int, blue: int):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(red, green, blue))
        self.setPalette(palette)


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self, graph):
        super().__init__()

        self.setWindowTitle("My App")

        layout = QGridLayout()
        for node in graph.nodes.data("agent"):
            [currency_one, currency_two] = node[1].wallet
            sat_one = currency_one / (currency_one + currency_two)
            sat_two = currency_two / (currency_one + currency_two)
            layout.addWidget(
                Color(int(255 * sat_one), 0, int(255 * sat_two)), node[0][0], node[0][1]
            )

        self.setFixedSize(QSize(400, 600))

        # Set the central widget of the Window.
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
