from models.multiple_solution.swarm_based.PFA import BasePFA, OPFA, LPFA, IPFA
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 50,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": whale_f1
}
pfa_paras = {
    "epoch": 1000,
    "pop_size": 100
}

## Run model
md = BasePFA(root_algo_paras=root_paras, pfa_paras=pfa_paras)
md._train__()

