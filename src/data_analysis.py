from typing import Dict
from sim.grid import gen_econ_network
from sim.config import exp

import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def scatter(assessments_dict: Dict):
    x = [val[0] for val in assessments_dict.values()]
    y = [val[1] for val in assessments_dict.values()]

    plt.scatter(x, y)
    plt.savefig(f"fig/plot-{time.time()}.png")


df: pd.DataFrame = pickle.load(open("sim_results-1719003096.8714561", "rb"))
df.set_index("timestep")
print(df.keys())
scatter(df.loc[0]["pricing_assessments"])
scatter(df.loc[49]["pricing_assessments"])
