from typing import Dict, Tuple, List, Any
from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import config_sim
from cadCAD.types import (
    Parameters,
    PolicyOutput,
    State,
    StateHistory,
    StateVariable,
    Substep,
)
import networkx as nx
import numpy as np
import numpy.typing as npt

from sim.grid import Agent, gen_econ_network, gen_random_assessments

eps: float = 10**-6

# ============================================================
# Policy functions
# ============================================================


def compute_pricing_assessment(
    _params: Parameters, substep: Substep, state_history: StateHistory, state: State
) -> PolicyOutput:
    return {"pricing_assessments": state["pricing_assessments"]}


def compute_inhereted_assessment(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: Dict[str, Any],
) -> PolicyOutput:

    grid: nx.DiGraph = state["grid"]
    inherited_assessments: Dict[Tuple[int, int], npt.NDArray[np.float64]] = {}
    best_vendors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for node, data in grid.nodes(data=True):
        customer: Agent = data["agent"]
        nodes_best_vendors: List[Tuple[int, int]] = []
        max_util_scale: float = -np.inf

        # Find best vendor among neighbors
        for neighbor in grid[node]:
            vendor: Agent = grid.nodes[neighbor]["agent"]

            util_scale: float = (
                customer.demand[neighbor]
                / vendor.price
                * np.dot(customer.wallet, state["pricing_assessments"][neighbor])
                # / np.linalg.norm(customer.wallet) # This line is constant factor, unneeded
            )

            if util_scale > max_util_scale:
                max_util_scale = util_scale
                nodes_best_vendors = [neighbor]
            elif max_util_scale - util_scale < eps:
                nodes_best_vendors.append(neighbor)

        # Add node data to policy dict
        # HACK: this could break if best vendors list is empty
        inherited_assessments[node] = state["pricing_assessments"][
            nodes_best_vendors[0]
        ]
        best_vendors[node] = nodes_best_vendors

    return {
        "inhereted_assessments": inherited_assessments,
        "best_vendors": best_vendors,
    }


# ============================================================
# State update functions
# ============================================================


def simulate_purchases(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: Dict[str, Any],
    _input: Dict[str, Any],
) -> Tuple[str, StateVariable]:
    grid = state["grid"].copy()

    # TODO: I want to iterate this in a random order
    for node, data in grid.nodes(data=True):
        customer: Agent = data["agent"]

        # randomly choose vendor to buy from
        vendor_idx = np.random.choice(len(_input["best_vendors"][node]))
        vendor_coords = _input["best_vendors"][node][vendor_idx]
        vendor: Agent = grid.nodes[vendor_coords]["agent"]

        # check if node has enough money and wants product more than cost
        required_payment_mag: float = vendor.price / np.dot(
            customer.wallet, _input["pricing_assessments"][vendor_coords]
        )

        can_pay: bool = bool(required_payment_mag < np.linalg.norm(customer.wallet))
        wants_product: bool = customer.demand[vendor_coords] > vendor.price * np.dot(
            customer.wallet, _input["inhereted_assessments"][node]
        ) / np.dot(customer.wallet, _input["pricing_assessments"][vendor_coords])

        if can_pay and wants_product:
            payment = customer.wallet * required_payment_mag
            customer.wallet -= payment
            vendor.wallet += payment

    return ("grid", grid)


def update_best_vendors(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: Dict[str, Any],
    _input: PolicyOutput,
) -> Tuple[str, StateVariable]:
    return ("best_vendors", _input["best_vendors"])


def update_inherited_assessments(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: Dict[str, Any],
    _input: PolicyOutput,
) -> Tuple[str, StateVariable]:
    return ("inherited_assessments", _input["inherited_assessments"])


def update_pricing_assessments(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: Dict[str, Any],
    _input: PolicyOutput,
) -> Tuple[str, StateVariable]:
    return ("pricing_assessments", _input["pricing_assessments"])


# ============================================================
# Sim configuration
# ============================================================

# TODO: move state creation code here
initial_graph = gen_econ_network()

initial_state = {
    "grid": initial_graph,
    "best_vendors": {},
    "inherited_assessments": gen_random_assessments(initial_graph),
    "pricing_assessments": gen_random_assessments(initial_graph),
}

psubs = [
    {
        "policies": {
            "pub_assessment": compute_pricing_assessment,
            "inhereted_assessment": compute_inhereted_assessment,
        },
        "variables": {"grid": simulate_purchases},
    }
]

sim_config = config_sim({"N": 2, "T": range(100)})

exp = Experiment()
exp.append_configs(
    model_id="first_order_iw",
    sim_configs=sim_config,
    initial_state=initial_state,
    partial_state_update_blocks=psubs,
)
