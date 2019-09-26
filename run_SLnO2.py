from models.multiple_solution.swarm_based.SLnO import SLnO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C29
}
woa_paras = {
    "epoch": 500,
    "pop_size": 100
}

## Run model
md = SLnO(root_algo_paras=root_paras, woa_paras=woa_paras)
md._train__()