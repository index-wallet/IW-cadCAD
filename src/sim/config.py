from typing import Callable, Dict, Tuple, List, Any
from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import config_sim
from cadCAD.types import (
    Parameters,
    PolicyOutput,
    StateHistory,
    StateVariable,
    Substep,
)
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.optimize
import pandas as pd
import pickle

from itertools import chain, combinations
import warnings

from sim.grid import Agent, gen_econ_network, gen_random_assessments

warnings.filterwarnings(
    "ignore", message="delta_grad == 0.0. Check if the approximated function is linear."
)
eps: float = 10**-6

# ============================================================
# Policy functions
# ============================================================


def compute_pricing_assessment(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: Dict[str, Any],
) -> PolicyOutput:

    grid: nx.DiGraph = state["grid"]
    pricing_assessments: Dict[Tuple[int, int], npt.NDArray] = {}
    # Fill if needed (on first iteration)
    best_vendors = state["best_vendors"]
    if best_vendors == {}:
        best_vendors = {
            node: get_best_vendors(grid, node, state["pricing_assessments"])
            for node in grid
        }
    print(f"Starting optimization, timestep{state["timestep"]}")
    for node in grid.nodes():
        customer_idx = grid[node]
        customers = [grid.nodes[idx]["agent"] for idx in customer_idx]
        customer_combos = powerset(
            zip(customer_idx, customers)
        )  # PERF: For highly connected topologies, this becomes inefficient
        me: Agent = grid.nodes[node]["agent"]

        def get_profit_func(
            customer_combo: List[Tuple[Tuple[int, int], Agent]],
        ) -> Callable[[npt.NDArray], float]:
            """Returns experienced utility when selling to every customer in combo list"""
            # PERF: This can definitely be optimized using np native functions

            def profit_func(assessment: npt.NDArray) -> float:
                profit: float = 0

                for customer_idx, customer in customer_combo:
                    pricing_assessments = state["pricing_assessments"][node]
                    if np.linalg.norm(pricing_assessments) == 0:
                        customer_revenue = 0
                    else:
                        customer_revenue = (
                            np.dot(
                                customer.wallet, state["inherited_assessments"][node]
                            )
                            / np.dot(customer.wallet, pricing_assessments)
                            / (1 + len(best_vendors[customer_idx]))
                        )
                        profit -= me.prod_cost

                    profit += me.price * customer_revenue

                # negative here b/c scipy only minimizes
                return -profit

            return profit_func

        max_profit = 0
        best_assessment = np.zeros_like(me.wallet)
        # best_combo = []

        # Find optimal assessment for every combo
        for combo in customer_combos:
            objective = get_profit_func(list(combo))

            constraint = scipy.optimize.LinearConstraint(
                np.array([customer.wallet for _, customer in combo]),
                lb=np.array(
                    [
                        me.price
                        * np.dot(customer.wallet, state["inherited_assessments"][node])
                        / (
                            customer.demand[node]
                            - customer.demand[best_vendors[idx][0]]
                            + grid.nodes[best_vendors[idx][0]]["agent"].price
                        )
                        for idx, customer in combo
                    ]
                ),
            )

            result = scipy.optimize.minimize(
                objective,
                state["inherited_assessments"][node],
                constraints=constraint,
                method="trust-constr",
            )

            profit: float = -result.fun
            assessment: npt.NDArray = result.x

            # Search optimal profit over every combo
            if profit > max_profit:
                max_profit = profit
                best_assessment = assessment
                # best_combo = combo

        pricing_assessments[node] = best_assessment
        # print(
        #     f"Node {node} makes profit {max_profit} by selling to {len(best_combo)} neighbors"
        # )

    return {"pricing_assessments": pricing_assessments}


def compute_inherited_assessment(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: Dict[str, Any],
) -> PolicyOutput:

    grid: nx.DiGraph = state["grid"]
    inherited_assessments: Dict[Tuple[int, int], npt.NDArray[np.float64]] = {}
    best_vendors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for node in grid.nodes():
        # Find best vendor among neighbors
        nodes_best_vendors = get_best_vendors(grid, node, state["pricing_assessments"])

        # Add node data to policy dict
        # HACK: this could break if best vendors list is empty
        inherited_assessments[node] = state["pricing_assessments"][
            nodes_best_vendors[0]
        ]
        best_vendors[node] = nodes_best_vendors

    return {
        "inherited_assessments": inherited_assessments,
        "best_vendors": best_vendors,
    }


def get_best_vendors(
    grid: nx.DiGraph, node, pricing_assessments
) -> List[Tuple[int, int]]:
    best_vendors: List[Tuple[int, int]] = []
    max_util_scale: float = -np.inf
    customer: Agent = grid.nodes[node]["agent"]

    for neighbor in grid[node]:
        vendor: Agent = grid.nodes[neighbor]["agent"]

        util_scale: float = (
            customer.demand[neighbor]
            / vendor.price
            * np.dot(customer.wallet, pricing_assessments[neighbor])
            # / np.linalg.norm(customer.wallet) # This line is constant factor, unneeded
        )

        if util_scale > max_util_scale:
            max_util_scale = util_scale
            best_vendors = [neighbor]
        elif max_util_scale - util_scale < eps:
            best_vendors.append(neighbor)

    return best_vendors


def powerset(iterable):
    "Excludes empty: powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


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
        denom = np.dot(customer.wallet, _input["pricing_assessments"][vendor_coords])
        required_payment_mag = 0
        effective_price = 0
        if denom == 0:
            required_payment_mag = np.inf
            effective_price = np.inf
        else:
            required_payment_mag: float = vendor.price / denom
            effective_price = (
                vendor.price
                * np.dot(customer.wallet, _input["inherited_assessments"][node])
                / denom
            )

        can_pay: bool = bool(required_payment_mag < np.linalg.norm(customer.wallet))
        wants_product: bool = customer.demand[vendor_coords] > effective_price

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

initial_graph = gen_econ_network()

initial_state = {
    "grid": initial_graph,
    "best_vendors": {},
    "inherited_assessments": gen_random_assessments(initial_graph),
    # NOTE: the price vendors charge is not variable, because they can accomplish the same
    # thing by changing the magnitude of their pricing assessments and leaving direction the same
    "pricing_assessments": gen_random_assessments(initial_graph),
}

psubs = [
    {
        "policies": {
            "pub_assessment": compute_pricing_assessment,
            "inherited_assessment": compute_inherited_assessment,
        },
        "variables": {
            "grid": simulate_purchases,
            "best_vendors": update_best_vendors,
            "inherited_assessments": update_inherited_assessments,
            "pricing_assessments": update_pricing_assessments,
        },
    }
]

sim_config = config_sim({"N": 1, "T": range(30)})


def exp(startfile: str | None = None):
    state = initial_state
    if startfile is not None:
        file = open(startfile, "rb")
        df = pickle.load(file)

        if type(df) is pd.Series:
            df["timestep"] = 0
            df = df.to_frame().T

        print(df.keys())
        state = {
            "grid": df.iloc[0]["grid"],
            "best_vendors": {},
            "inherited_assessments": df.iloc[0]["inherited_assessments"],
            "pricing_assessments": df.iloc[0]["pricing_assessments"],
        }

    xp = Experiment()
    xp.append_configs(
        model_id="first_order_iw",
        sim_configs=sim_config,
        initial_state=state,
        partial_state_update_blocks=psubs,
    )

    return xp
