from cadCAD.engine import ExecutionContext, ExecutionMode, Executor
import pandas as pd
import pickle
from datetime import datetime
import os
import subprocess

from sim.config import exp

os.chdir("/home/bgould/dev/index-wallets/IW-cadCAD")
conf_file: str | None = (
    "sim_results/99f8fb74a11cbfcf345671cf823f6af5ef1700c9/2024-06-27 14:32:14.sim"
)
conf_file = None

exec_context = ExecutionContext(context=ExecutionMode().single_mode)
# exec_context = ExecutionContext()
run = Executor(
    exec_context=exec_context,
    configs=exp(conf_file).configs,
)

(system_events, tensor_field, sessions) = run.execute()
df = pd.DataFrame(system_events)

try:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    commit_hash = result.stdout.decode("utf-8").strip()
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e.stderr.decode('utf-8')}")
    commit_hash = "no_repo_found"

save_dir: str = f"sim_results/{commit_hash}"
os.makedirs(save_dir, exist_ok=True)

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

picklefile = open(f"{save_dir}/{timestamp}.sim", "wb")
if conf_file is not None:
    picklefile = open(conf_file.replace(".sim", ".sim2"), "wb")

pickle.dump(df, picklefile)
