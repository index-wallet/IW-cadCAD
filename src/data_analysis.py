import os
from typing import Dict, List

import pickle
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx


def scatter(assessments_dict: Dict, savepath: str):
    x = [val[0] for val in assessments_dict.values()]
    y = [val[1] for val in assessments_dict.values()]

    plt.figure()
    plt.scatter(x, y)
    plt.gca().set_title("Pricing Assessments")
    plt.savefig(savepath)


def lineplot(vals: List[Dict], savepath: str):
    df = pd.DataFrame(vals)
    ax: matplotlib.axes.Axes = df.plot.line()
    ax.get_legend().remove()
    ax.set_title("Wealth over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Currency")
    ax.get_figure().savefig(savepath)


def report(filename: str):
    save_dir: str = f"fig/{filename}/"
    os.makedirs(save_dir, exist_ok=True)
    df: pd.DataFrame = pickle.load(open(filename, "rb"))
    df.set_index("timestep")

    grids = df["grid"]
    currency_one_list, currency_two_list = [], []

    for grid in grids:
        agents = nx.get_node_attributes(grid, "agent")
        currency_one_list.append(
            {node: agents[node].wallet[0] for node in agents.keys()}
        )
        currency_two_list.append(
            {node: agents[node].wallet[1] for node in agents.keys()}
        )

    scatter(df.loc[0]["pricing_assessments"], save_dir + "initial_assessments.png")
    scatter(df.iloc[-1]["pricing_assessments"], save_dir + "final_assessments.png")
    lineplot(currency_one_list, save_dir + "currency_one.png")
    lineplot(currency_two_list, save_dir + "currency_two.png")


report("sim_results/brute_force_powerset_search/sim_results-1719003096.8714561")
