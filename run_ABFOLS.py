from models.multiple_solution.swarm_based.BFO import ABFOLS
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": C4
}
abfols_paras = {
    "epoch": 500,
    "pop_size": 100,
    "Ci": [0.1, 0.001],         # C_s (start), C_e (end)  -=> step size # step size in BFO
    "Ped": 0.25,                  # p_eliminate
    "Ns": 4,                      # swim_length
    "N_minmax": [3, 40],          # (Dead threshold value, split threshold value) -> N_adapt, N_split
}

## Run model
md = ABFOLS(root_algo_paras=root_paras, abfols_paras=abfols_paras)
md._train__()

