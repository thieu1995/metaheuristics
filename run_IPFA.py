from models.multiple_solution.swarm_based.PFA import BasePFA, OPFA, LPFA, IPFA
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C29
}
pfa_paras = {
    "epoch": 500,
    "pop_size": 100
}

## Run model
md = IPFA(root_algo_paras=root_paras, pfa_paras=pfa_paras)
md._train__()

