from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QGridLayout,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QPushButton,
    QSlider,
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
        self.setFixedSize(QSize(400, 600))

        # Construct UI pieces
        main_layout = QVBoxLayout()
        agent_grid_layout = self.__agent_grid__(graph)
        timeline_slider = QSlider(Qt.Orientation.Horizontal)
        timeline_slider.setRange(0, 50)
        timeline_slider.setSingleStep(1)

        # Organize UI
        main_layout.addLayout(agent_grid_layout)
        main_layout.addWidget(timeline_slider)
        main_layout.addLayout(self.__sim_btns__())

        # Set the central widget of the Window.
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def __agent_grid__(self, graph) -> QGridLayout:
        agent_grid_layout = QGridLayout()
        for node, agent in graph.nodes.data("agent"):
            [currency_one, currency_two] = agent.wallet
            sat_one = currency_one / (currency_one + currency_two)
            sat_two = currency_two / (currency_one + currency_two)
            agent_grid_layout.addWidget(
                Color(int(255 * sat_one), 0, int(255 * sat_two)), node[0], node[1]
            )

        return agent_grid_layout

    def __sim_btns__(self) -> QHBoxLayout:
        open_btn = QPushButton("Open Sim File")
        open_btn.clicked.connect(self.openFile)

        save_btn = QPushButton("Fork this timestep")
        save_btn.clicked.connect(self.saveFile)

        layout = QHBoxLayout()
        layout.addWidget(open_btn)
        layout.addWidget(save_btn)

        return layout

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sim to Open",
            "",
            "All Files (*)",
        )
        if fileName:
            print(f"Loading sim: {fileName}")

    def saveFile(self):
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Select Fork Save Location",
            "",
            "All Files (*)",
        )
        if fileName:
            print(f"Saving fork to: {fileName}")
