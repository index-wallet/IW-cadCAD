import os
from typing import Dict, List

import pickle
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def scatter(assessments_dict: Dict, savepath: str):
    x = [val[0] for val in assessments_dict.values()]
    y = [val[1] for val in assessments_dict.values()]

    plt.figure()
    plt.scatter(x, y)
    plt.gca().set_title("Pricing Assessments")
    plt.savefig(savepath)
    plt.close()


def lineplot(vals: List[Dict], savepath: str):
    df = pd.DataFrame(vals)
    ax: matplotlib.axes.Axes = df.plot.line()
    ax.get_legend().remove()
    ax.set_title("Wealth over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Currency")

    fig = ax.get_figure()
    if fig is not None:
        fig.savefig(savepath)
    plt.close()


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

    norm_init_asmt = {
        node: asmt / grids[0].nodes[node]["agent"].price
        for node, asmt in df.loc[0]["pricing_assessments"].items()
    }
    norm_final_asmt = {
        node: asmt / grids[0].nodes[node]["agent"].price
        for node, asmt in df.iloc[-1]["pricing_assessments"].items()
    }

    # NOTE: Normalize here by the price scalar to account for ability to change
    # price by assessment magnitude
    scatter(norm_init_asmt, save_dir + "initial_assessments.png")
    scatter(norm_final_asmt, save_dir + "final_assessments.png")
    lineplot(currency_one_list, save_dir + "currency_one.png")
    lineplot(currency_two_list, save_dir + "currency_two.png")


directory = "sim_results/99f8fb74a11cbfcf345671cf823f6af5ef1700c9"
for file in os.listdir(directory):
    report(os.path.join(directory, file))
