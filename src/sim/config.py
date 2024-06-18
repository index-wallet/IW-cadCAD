from typing import Tuple
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

from sim.grid import gen_econ_network


def compute_pub_assessment(
    _params: Parameters, substep: Substep, state_history: StateHistory, state: State
) -> PolicyOutput:
    return {}


def compute_inhereted_assessment(
    _params: Parameters, substep: Substep, state_history: StateHistory, state: State
) -> PolicyOutput:
    return {}


def simulate_purchases(
    _params: Parameters,
    substep: Substep,
    state_history: StateHistory,
    state: State,
    _input: PolicyOutput,
) -> Tuple[str, StateVariable]:
    return ("", 0)


initial_state = {"grid": gen_econ_network()}

psubs = [
    {
        "policies": {
            "pub_assessment": compute_pub_assessment,
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
