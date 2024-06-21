import sys

from cadCAD.engine import ExecutionContext, ExecutionMode, Executor
import pandas as pd
import pickle
import time

from PyQt6.QtWidgets import QApplication
from sim.grid import gen_econ_network
from sim.config import exp
from ui.window import MainWindow

# exec_context = ExecutionContext(context=ExecutionMode().single_mode)
exec_context = ExecutionContext()
run = Executor(exec_context=exec_context, configs=exp.configs)

(system_events, tensor_field, sessions) = run.execute()
df = pd.DataFrame(system_events)

picklefile = open(f"sim_results-{time.time()}", "wb")
pickle.dump(df, picklefile)

app = QApplication(sys.argv)
window = MainWindow(gen_econ_network())
window.show()

app.exec()
