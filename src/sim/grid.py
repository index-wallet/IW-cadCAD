from typing import Dict, List, Tuple

import networkx as nx
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


class Agent:
    def __init__(self, neighbors: List[Tuple[int, int]]) -> None:
        self.wallet: npt.NDArray[np.float64] = np.random.rand(num_currencies)

        self.price: float = np.random.random() * price_range[1] + price_range[0]
        # HACK: this feels hacky, but I can't use agents as keys if I want to construct them all at the same time
        self.demand: Dict[Tuple[int, int], float] = {
            neighbor: np.random.random() * demand_range[1] + demand_range[0]
            for neighbor in neighbors
        }

        self.inherited_assessment: npt.NDArray[np.float64] = (
            np.random.rand(num_currencies) * valuation_range[1] + valuation_range[0]
        )
        self.pricing_assessment: npt.NDArray[np.float64] = (
            np.random.rand(num_currencies) * valuation_range[1] + valuation_range[0]
        )

    def __str__(self) -> str:
        return f"Wallet: {self.wallet}\nPrice: {self.price}\nDemand: {self.demand}\nInherited Valuation: {self.inherited_assessment}\nPricing Valuation:{self.pricing_assessment}"


def gen_econ_network() -> nx.DiGraph:
    # Create grid graph
    graph = nx.grid_2d_graph(
        grid_size, grid_size, periodic=wraparound, create_using=nx.DiGraph
    )

    # Create agent at each node
    node_data = {node: {"agent": Agent(graph[node])} for node in graph}
    nx.set_node_attributes(graph, node_data)

    # Dropout
    to_remove = []
    for edge in graph.edges:
        if np.random.random() > edge_prob:
            to_remove.append(edge)
    graph.remove_edges_from(to_remove)

    return graph
