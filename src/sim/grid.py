from typing import Dict, List, Tuple, Literal
import logging

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
    topology_type,
    vendor_customer_ratio
)


class Agent:
    def __init__(self, neighbors: List[Tuple[int, int]] | None = None, type: Literal["both", "vendor", "customer"] = "both"):
        ## Type is only used for vendor_customer topology
        ## All agents are prosumers, this merely dictates who can trade with who
        ## Neighbors are used for grid topology, I couldn't figure out a way around the hack that used by the original code and messing with it broke things so I left it as is
        ## But vendor_customer can just do it after the fact 
        self.wallet: npt.NDArray[np.float64] = np.random.rand(num_currencies)

        self.price: float = np.random.random() * price_range[1] + price_range[0]
        self.prod_cost = self.price / 10
        self.type = type
        self.public_good_util_scales = np.zeros(num_currencies)
        
        if neighbors is not None:
            # HACK: this feels hacky, but I can't use agents as keys if I want to construct them all at the same time
            self.demand: Dict[Tuple[int, int], float] = {
                neighbor: np.random.random() * demand_range[1] + demand_range[0]
                for neighbor in neighbors
            }
        else:
            self.demand = {}

    def initialize_demand(self, neighbors: List[Tuple[int, int]]):
        self.demand = {
            neighbor: np.random.random() * demand_range[1] + demand_range[0]
            for neighbor in neighbors
        }

    def __str__(self) -> str:
        return f"Wallet: {self.wallet}\nPrice: {self.price}\nDemand: {self.demand}\nType: {self.type}"


def gen_econ_network() -> nx.DiGraph:
    logging.debug(f"Generating {topology_type} topology")
    if topology_type == "grid":
        graph = get_graph_topology()
    elif topology_type == "vendor_customer":
        graph = get_vendor_customer_topology()
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    return graph

def get_graph_topology() -> nx.DiGraph:
    # Create grid graph
    graph = nx.grid_2d_graph(
        grid_size, grid_size, periodic=wraparound, create_using=nx.DiGraph
    )

    # Create agent at each node
    node_data = {node: {"agent": Agent(neighbors=graph[node])} for node in graph}
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

def get_vendor_customer_topology() -> nx.DiGraph:

    if vendor_customer_ratio < 1:
        raise ValueError(f"vendor_customer_ratio must be at least 1, got {vendor_customer_ratio}")

    graph = nx.DiGraph()
    total_nodes = grid_size * grid_size
    
    ## Calculate the number of vendors and customers
    num_vendors = max(1, total_nodes // (vendor_customer_ratio + 1))
    num_customers = total_nodes - num_vendors
    actual_ratio = num_customers / num_vendors
    if abs(actual_ratio - vendor_customer_ratio) > 0.1 * vendor_customer_ratio:
        logging.warning(f"Actual customer-to-vendor ratio ({actual_ratio:.2f}) differs significantly "
                       f"from requested ratio ({vendor_customer_ratio:.2f})")
    
    ## Create agents and assign to nodes
    vendors = set(np.random.choice(total_nodes, num_vendors, replace=False))
    node_data = {}
    for i in range(total_nodes):
        node = (i // grid_size, i % grid_size)  ## 2d convert
        agent_type = "vendor" if i in vendors else "customer"
        node_data[node] = {"agent": Agent(type=agent_type)}
        graph.add_node(node, **node_data[node])
   
    vendor_nodes = [node for node in graph if graph.nodes[node]["agent"].type == "vendor"]
    customer_nodes = [node for node in graph if graph.nodes[node]["agent"].type == "customer"]
   
    edges = [(v, c) for v in vendor_nodes for c in customer_nodes]
    graph.add_edges_from(edges)
    graph.add_edges_from((c, v) for (v, c) in edges)

    ## Initialize demands for each agent based on final graph structure
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        graph.nodes[node]['agent'].initialize_demand(neighbors)
    
    logging.debug("Created graph with {num_vendors} vendors and {num_customers} customers")
    logging.debug(f"Total edges created: {graph.number_of_edges()}")
    
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
