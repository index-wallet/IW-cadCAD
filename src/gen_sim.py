from cadCAD.engine import ExecutionContext, ExecutionMode, Executor
import pandas as pd
import pickle
from datetime import datetime
import os

from sim.config import exp
from util.utils import get_latest_commit_hash

# Pretty sure this just overrides value with None, gonna leave it for now
conf_file: str | None = (
    "sim_results/99f8fb74a11cbfcf345671cf823f6af5ef1700c9/2024-06-27 14:32:14.sim"
)
conf_file = None

# switch to single threaded mode here for debugging
exec_context = ExecutionContext(context=ExecutionMode().single_mode)
# os.chdir("/home/bgould/dev/index-wallets/IW-cadCAD")
# exec_context = ExecutionContext()

run = Executor(
    exec_context=exec_context,
    configs=exp(conf_file).configs,
)

(system_events, tensor_field, sessions) = run.execute()
df = pd.DataFrame(system_events)

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