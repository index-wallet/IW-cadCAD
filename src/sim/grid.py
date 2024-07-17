from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt

grid_size: int = 10
edge_prob: float = 0.8
seed: np.random.RandomState = np.random.RandomState(0)
wraparound: bool = True

num_currencies: int = 1

# Each of these ranges is [min, width] for a uniform sample
demand_range: List[float] = [0.5, 1]
valuation_range: List[float] = [0.5, 1]
price_range: List[float] = [0.1, 0.9]


class Agent:
    def __init__(self, neighbors: List[Tuple[int, int]]) -> None:
        self.wallet: npt.NDArray[np.float64] = np.random.rand(num_currencies)

        self.price: float = np.random.random() * price_range[1] + price_range[0]
        self.prod_cost = self.price / 10
        # HACK: this feels hacky, but I can't use agents as keys if I want to construct them all at the same time
        self.demand: Dict[Tuple[int, int], float] = {
            neighbor: np.random.random() * demand_range[1] + demand_range[0]
            for neighbor in neighbors
        }

        # scalar multiplier for public good benefit
        # worst case: no one gets any utility from donations
        self.public_good_util_scales = np.zeros(num_currencies)

    def __str__(self) -> str:
        return f"Wallet: {self.wallet}\nPrice: {self.price}\nDemand: {self.demand}"


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
    orphan_nodes = [node for node, degree in graph.degree() if degree == 0]
    for orphan in orphan_nodes:
        neighbor = (orphan[0] + 1 % grid_size, orphan[1])
        graph.add_edges_from([(orphan, neighbor)])

    return graph


def gen_random_assessments(graph: nx.DiGraph):
    return {
        node: np.random.rand(num_currencies) * valuation_range[1] + valuation_range[0]
        for node in graph.nodes
    }


def public_good_util(previous_donations: float, new_donations: float) -> float:
    return new_donations * 0.5 / np.log(previous_donations / 10 + 1.1)


def donation_currency_reward(
    pricing_assessments: Dict[Tuple[int, int], npt.NDArray],
    good_index: int,
    previous_donations: float,
    new_donations: float,
) -> float:
    good_currency_asmts: npt.NDArray = np.array(
        [asmt[good_index] for asmt in pricing_assessments.values()]
    )

    # Weight random choice towards higher valuations
    # TODO: Is this really what we want? It actually skews towards giving less currency back
    # HACK: We could have problems here if we choose a 0 ref_asmt. Theoretically shouldn't happen
    ref_asmt = np.random.choice(
        good_currency_asmts,
        p=(good_currency_asmts / np.sum(good_currency_asmts)),
    )

    return public_good_util(previous_donations, new_donations) / ref_asmt
