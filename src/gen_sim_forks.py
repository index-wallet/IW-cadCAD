import numpy as np
import networkx as nx
import pandas as pd
import pickle

from util.utils import get_latest_sim


def find_extreme_nodes(df: pd.DataFrame):
    grid = df.iloc[-1]["grid"]
    agents = nx.get_node_attributes(grid, "agent")

    final_wealth = {node: agents[node].wallet for node in agents.keys()}
    rich_node = max(final_wealth, key=lambda k: np.linalg.norm(final_wealth[k]))
    poor_node = min(final_wealth, key=lambda k: np.linalg.norm(final_wealth[k]))

    return (rich_node, poor_node)


simfile = get_latest_sim()

df = pickle.load(open(simfile, "rb"))

if type(df) is pd.Series:
    df["timestep"] = 0
    df = df.to_frame().T

df.set_index("timestep")

rich_node, poor_node = find_extreme_nodes(df)
bump_factor = np.array([2, 1])

rich_asmt_bump = df.copy()
rich_asmt_bump.iloc[0]["pricing_assessments"][rich_node] *= bump_factor
rich_asmt_bump_file = simfile.replace(".sim", "-FORK_rich_node_asmt_bump.state")
pickle.dump(rich_asmt_bump, open(rich_asmt_bump_file, "wb"))

poor_asmt_bump = df.copy()
poor_asmt_bump.iloc[0]["pricing_assessments"][poor_node] *= bump_factor
poor_asmt_bump_file = simfile.replace(".sim", "-FORK_poor_node_asmt_bump.state")
pickle.dump(poor_asmt_bump, open(poor_asmt_bump_file, "wb"))

rich_wallet_bump = df.copy()
rich_wallet_bump.iloc[0]["grid"].nodes[rich_node]["agent"].wallet *= bump_factor
rich_wallet_bump_file = simfile.replace(".sim", "-FORK_rich_node_wallet_bump.state")
pickle.dump(rich_wallet_bump, open(rich_wallet_bump_file, "wb"))

poor_wallet_bump = df.copy()
poor_wallet_bump.iloc[0]["grid"].nodes[poor_node]["agent"].wallet *= bump_factor
poor_wallet_bump_file = simfile.replace(".sim", "-FORK_poor_node_wallet_bump.state")
pickle.dump(poor_wallet_bump, open(poor_wallet_bump_file, "wb"))
