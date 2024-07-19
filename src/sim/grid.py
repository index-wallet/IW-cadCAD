from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt

from sim.params import (
    grid_size,
    edge_prob,
    wraparound,
    num_currencies,
    demand_range,
    valuation_range,
    price_range,
    initial_donation_reward_amount,
)


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


def find_public_good_owners(graph: nx.DiGraph) -> List[Tuple[int, int]]:
    """
    Find the node in the graph with the highest benefit scalar for each good.
    We assume that these nodes represent the owner of the company working on
    that good, and use their assessments to value donations.

    Args:
        graph: The network of agents in the sim

    Returns: A list of the coordinates of the owners of each good, in the same
    order as simulation currencies
    """

    data = graph.nodes.data("agent")
    util_scales = np.array([agent.public_good_util_scales for _, agent in data])
    node_coords = [node for node, _ in data]
    owner_idx = np.argmax(util_scales, axis=0)
    owner_coords = [node_coords[i] for i in owner_idx]

    return owner_coords


def public_good_util(
    previous_donations: npt.NDArray,
    new_donations: npt.NDArray,
    util_assessments: npt.NDArray,
) -> npt.NDArray:
    """
    Calculates the utility received by agents through a public good,
    given the amount of donations it has received. Utility increases
    sub-linearly with total donations.

    Args:
        previous_donations: 2D array, giving the wallet of previous donations for each good
        new_donations: 2D array, for each good, the amount of each currency donated to it by this agent this timestep
        util_assessments: array of the value of each currency to the owner of the public good

    Returns: Utility caused by public good at current level of donation
    """
    # convert donations to benefit for public good owner
    prev_donation_util = np.array(
        [
            np.dot(donation, util_asmt)
            for donation, util_asmt in zip(previous_donations, util_assessments)
        ]
    )
    new_donation_util = np.array(
        [
            np.dot(donation, util_asmt)
            for donation, util_asmt in zip(new_donations, util_assessments)
        ]
    )

    # Currently, we assume that the success of a good is logarithmic in donation size
    return np.log(prev_donation_util + new_donation_util + 1)


def donation_currency_reward(
    previous_donations: npt.NDArray,
    new_donations: npt.NDArray,
    util_assessments: npt.NDArray,
) -> npt.NDArray:
    """
    Gets the amount of currency rewarded in return for the given donation.
    Reward value decreases with total donations made

    Args:
        previous_donations: 2D array, giving the wallet of previous donations for each good
        new_donations: 2D array, for each good, the amount of each currency donated to it by this agent this timestep
        util_assessments: array of the value of each currency to the owner of the public good

    Returns: Array giving the size of the reward from each good caused by this donation
    """
    # convert donations to benefit for public good owner
    prev_donation_util = np.array(
        [
            np.dot(donation, util_asmt)
            for donation, util_asmt in zip(previous_donations, util_assessments)
        ]
    )
    new_donation_util = np.array(
        [
            np.dot(donation, util_asmt)
            for donation, util_asmt in zip(new_donations, util_assessments)
        ]
    )

    # Currently, we use a hyperbolic donation reward scaling function
    # TODO: mix currency reward between original donation and good's currency
    return new_donation_util / (prev_donation_util + 1 / initial_donation_reward_amount)
