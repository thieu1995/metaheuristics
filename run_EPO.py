from models.multiple_solution.swarm_based.EPO import BaseEPO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
epo_paras = {
    "epoch": 500,
    "pop_size": 100
}

## Run model
md = BaseEPO(root_algo_paras=root_paras, epo_paras=epo_paras)
md._train__()

