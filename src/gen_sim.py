from cadCAD.engine import ExecutionContext, Executor
import pandas as pd
import pickle
import time

from sim.config import exp

# exec_context = ExecutionContext(context=ExecutionMode().single_mode)
exec_context = ExecutionContext()
run = Executor(exec_context=exec_context, configs=exp().configs)

(system_events, tensor_field, sessions) = run.execute()
df = pd.DataFrame(system_events)

picklefile = open(f"sim_results-{time.time()}", "wb")
pickle.dump(df, picklefile)
