from typing import Dict, Tuple, List
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

from sim.grid import Agent, gen_econ_network

eps: float = 10**-6


def compute_pricing_assessment(
    _params: Parameters, substep: Substep, state_history: StateHistory, state: State
) -> PolicyOutput:
    return {"pricing_assessments": state["pricing_assessments"]}


def compute_inhereted_assessment(
    _params: Parameters, substep: Substep, state_history: StateHistory, state: State
) -> PolicyOutput:

    grid: nx.DiGraph = state["grid"]
    inherited_assessments: Dict[Tuple[int, int], npt.NDArray[np.float64]] = {}
    best_vendors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for node in grid.nodes:
        customer: Agent = node["agent"]
        nodes_best_vendors: List[Tuple[int, int]] = []
        max_util_scale: float = 0

        # Find best vendor among neighbors
        for neighbor in grid[node]:
            vendor: Agent = neighbor["agent"]

            util_scale: float = (
                customer.demand[neighbor]
                / vendor.price
                / np.linalg.norm(customer.wallet)
                * np.dot(customer.wallet, vendor.pricing_assessment)
            )

            if util_scale > max_util_scale:
                max_util_scale = util_scale
                nodes_best_vendors = [neighbor]
            elif max_util_scale - util_scale < eps:
                nodes_best_vendors.append(neighbor)

        # Add node data to policy dict
        # HACK: this could break if best vendors list is empty
        inherited_assessments[node] = grid[nodes_best_vendors[0]][
            "agent"
        ].pricing_assessment
        best_vendors[node] = nodes_best_vendors

    return {
        "inhereted_assessments": inherited_assessments,
        "best_vendors": best_vendors,
    }


def simulate_purchases(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: State,
    _input: PolicyOutput,
) -> Tuple[str, StateVariable]:
    return ("", 0)


# TODO: move state creation code here
initial_state = {
    "grid": gen_econ_network(),
    "best_vendors": {},
    "inherited_assessments": {},
    "pricing_assessments": {},
}

psubs = [
    {
        "policies": {
            "pub_assessment": compute_pricing_assessment,
            "inhereted_assessment": compute_inhereted_assessment,
        },
        "variables": {"wallets": simulate_purchases},
    }
]

sim_config = config_sim({"N": 2, "T": range(100), "M": {}})

exp = Experiment()
exp.append_model(
    model_id="first_order_iw",
    sim_configs=sim_config,
    initial_state=initial_state,
    partial_state_update_blocks=psubs,
)
