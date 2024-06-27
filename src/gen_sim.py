from cadCAD.engine import ExecutionContext, Executor
import pandas as pd
import pickle
from datetime import datetime
import os, subprocess

from sim.config import exp

# exec_context = ExecutionContext(context=ExecutionMode().single_mode)
exec_context = ExecutionContext()
run = Executor(exec_context=exec_context, configs=exp().configs)

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
pickle.dump(df, picklefile)
