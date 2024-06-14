import sys
from typing import List

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

grid_size: int = 10
edge_prob: float = 0.8
seed: np.random.RandomState = np.random.RandomState(0)
wraparound: bool = True

num_currencies: int = 2

# Each of these ranges is [min, width] for a uniform sample
demand_range: List[float] = [0.5, 1]
valuation_range: List[float] = [0.5, 1]
price_range: List[float] = [0.1, 0.9]

[UP, RIGHT, DOWN, LEFT] = [0, 1, 2, 3]


class Agent:
    def __init__(self) -> None:
        self.wallet: npt.NDArray[np.float64] = np.random.rand(num_currencies)

        self.price: float = np.random.random() * price_range[1] + price_range[0]
        self.demand: npt.NDArray[np.float64] = (
            np.random.rand(4) * demand_range[1] + demand_range[0]
        )  # 4 here is number of potential neighbors, maybe use dict for more flexibility

        self.inherited_valuation: npt.NDArray[np.float64] = (
            np.random.rand(num_currencies) * valuation_range[1] + valuation_range[0]
        )
        self.pricing_valuation: npt.NDArray[np.float64] = (
            np.random.rand(num_currencies) * valuation_range[1] + valuation_range[0]
        )

    def __str__(self) -> str:
        return f"Wallet: {self.wallet}\nPrice: {self.price}\nDemand: {self.demand}\nInherited Valuation: {self.inherited_valuation}\nPricing Valuation:{self.pricing_valuation}"


# Generate network topology
graph = nx.grid_2d_graph(
    grid_size, grid_size, periodic=wraparound, create_using=nx.DiGraph
)
node_data = {node: {"agent": Agent()} for node in graph}
nx.set_node_attributes(graph, node_data)

to_remove = []
for edge in graph.edges:
    if np.random.random() > edge_prob:
        to_remove.append(edge)
graph.remove_edges_from(to_remove)

# nx.draw_networkx(graph)
# plt.show()


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press Me!")

        self.setFixedSize(QSize(400, 300))

        # Set the central widget of the Window.
        self.setCentralWidget(button)


# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec()
