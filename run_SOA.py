from models.multiple_solution.swarm_based.SOA import BaseSOA, OriginalSOA
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": C1
}
soa_paras = {
    "epoch": 1000,
    "pop_size": 100,
}

## Run model
md = BaseSOA(root_algo_paras=root_paras, soa_paras=soa_paras)
md._train__()

