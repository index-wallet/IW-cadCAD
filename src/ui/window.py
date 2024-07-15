import pickle
from typing import Tuple
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QGridLayout,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QPushButton,
    QSlider,
)

import pandas as pd
import numpy as np


class AgentWidget(QWidget):

    def __init__(self, agent):
        super(AgentWidget, self).__init__()
        self.setAutoFillBackground(True)
        self.update_agent(agent)

    def update_agent(self, agent):
        self.agent = agent

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(*self.__calc_color__()))
        self.setPalette(palette)

    def __calc_color__(self) -> Tuple[int, int, int]:
        sum: float = np.sum(self.agent.wallet)
        ratios = [currency / sum for currency in self.agent.wallet]

        # TODO: display this in a more principled manner
        return (int(255 * ratios[0]), 0, int(255 * ratios[-1]))


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self, simfile: str):
        super().__init__()

        self.setWindowTitle("My App")
        self.setFixedSize(QSize(400, 600))

        self.active_sim_file = simfile
        self.active_sim_data = self.__load_sim_file__(self.active_sim_file)
        self.timestep = 0
        self.avg_asmt = np.array([0.0, 0.0])

        # Construct UI pieces
        main_layout = QVBoxLayout()
        self.agent_grid_layout = self.__agent_grid__(
            self.active_sim_data.iloc[0]["grid"]
        )
        self.avg_asmt_lbl = QLabel(f"Average Valuations: Red, Blue = {self.avg_asmt}")

        # Organize UI
        main_layout.addLayout(self.agent_grid_layout)
        main_layout.addWidget(self.avg_asmt_lbl)
        main_layout.addWidget(self.__timeline_slider__())
        main_layout.addLayout(self.__sim_btns__())

        # Set the central widget of the Window.
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def __agent_grid__(self, graph) -> QGridLayout:
        agent_grid_layout = QGridLayout()
        for node, agent in graph.nodes.data("agent"):
            agent_grid_layout.addWidget(AgentWidget(agent), node[0], node[1])

        return agent_grid_layout

    def __timeline_slider__(self) -> QSlider:
        timeline_slider = QSlider(Qt.Orientation.Horizontal)
        timeline_slider.setRange(0, len(self.active_sim_data) - 1)
        timeline_slider.setSingleStep(1)
        timeline_slider.valueChanged.connect(self.show_timestep)

        return timeline_slider

    def show_timestep(self, timestep):
        self.timestep = timestep
        self.avg_asmt = self.calc_average_valuations()
        grid = self.active_sim_data.iloc[timestep]["grid"]

        for node, agent in grid.nodes.data("agent"):
            agent_widget: AgentWidget = self.agent_grid_layout.itemAtPosition(
                *node
            ).widget()
            agent_widget.update_agent(agent)
        self.avg_asmt_lbl.setText(f"Average Valuations: Red, Blue = {self.avg_asmt}")

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

            self.active_sim_file = fileName
            self.active_sim_data = self.__load_sim_file__(self.active_sim_file)
            self.show_timestep(0)

    def saveFile(self):
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Select Fork Save Location",
            "",
            "All Files (*)",
        )
        if fileName:
            print(f"Saving fork to: {fileName}")

            file = open(fileName, "wb")
            pickle.dump(self.active_sim_data.iloc[self.timestep], file)

    def __load_sim_file__(self, filepath: str) -> pd.DataFrame:
        file = open(filepath, "rb")
        df: pd.DataFrame = pickle.load(file)

        if type(df) is pd.Series:
            df["timestep"] = 0
            df = df.to_frame().T

        df.set_index("timestep")
        return df

    def calc_average_valuations(self):
        asmts = self.active_sim_data.iloc[self.timestep]["pricing_assessments"]
        valuation = np.array([0.0, 0.0])

        for _, asmt in asmts.items():
            valuation += asmt

        valuation /= 100
        return valuation
