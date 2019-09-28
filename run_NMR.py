from models.multiple_solution.swarm_based.NMR import BaseNMR, LevyNMR
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
nmr_paras = {
    "pop_size": 100,
    "epoch": 500,
    "bp": 0.75,      # breeding probability
}

## Run model
md = LevyNMR(root_algo_paras=root_paras, nmr_paras=nmr_paras)
md._train__()

