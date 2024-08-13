from cadCAD.engine import ExecutionContext, ExecutionMode, Executor
import pandas as pd
import pickle
from datetime import datetime
import os
import logging

from sim.config import exp
from util.utils import get_latest_commit_hash

from sim.params import is_debug

conf_file: str | None

## Uncomment to load a specific configuration file
## conf_file = ("sim_results/99f8fb74a11cbfcf345671cf823f6af5ef1700c9/2024-06-27 14:32:14.sim")
conf_file = None

if is_debug:
    exec_context = ExecutionContext(context=ExecutionMode().single_mode)

    ## Setup logging
    logging.basicConfig(level=logging.DEBUG, 
                        format='[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    if not any(isinstance(handler, logging.StreamHandler) for handler in logging.getLogger('').handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

else:
    exec_context = ExecutionContext(context=ExecutionMode().local_mode)

run = Executor(
    exec_context=exec_context,
    configs=exp(conf_file).configs,
)

(system_events, tensor_field, sessions) = run.execute()
df = pd.DataFrame(system_events)

def add_edge_data(row):
    """ Add edge data to the dataframe """
    grid = row['grid']
    edges = list(grid.edges())
    row['edges'] = edges
    return row

df = df.apply(add_edge_data, axis=1)

commit_hash = get_latest_commit_hash()

save_dir: str = f"sim_results/{commit_hash}"
os.makedirs(save_dir, exist_ok=True)

# Windows doesn't like colons in filenames
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

with open(f"{save_dir}/{timestamp}.sim", "wb") as picklefile:
    if conf_file is not None:
        with open(conf_file.replace(".sim", ".sim-fork"), "wb") as forkfile:
            pickle.dump(df, forkfile)
    else:
        pickle.dump(df, picklefile)