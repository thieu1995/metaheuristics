from models.multiple_solution.swarm_based.SLnO import MSLnO
from utils.FunctionUtil import *

## Setting parameters`
root_paras = {
    "problem_size": 30,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C26
}
woa_paras = {
    "epoch": 500,
    "pop_size": 100
}

## Run model
md = MSLnO(root_algo_paras=root_paras, woa_paras=woa_paras)
md._train__()
