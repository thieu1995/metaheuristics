from models.multiple_solution.swarm_based.BMO import BaseBMO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
bmo_paras = {
    "epoch": 500,
    "pop_size": 100,
    "bm_teams": 10
}

## Run model
md = BaseBMO(root_algo_paras=root_paras, bmo_paras=bmo_paras)
md._train__()

