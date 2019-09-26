from models.multiple_solution.swarm_based.SOA import BaseSOA, OriginalSOA
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C29
}
soa_paras = {
    "epoch": 500,
    "pop_size": 100,
}

## Run model
md = BaseSOA(root_algo_paras=root_paras, soa_paras=soa_paras)
md._train__()

