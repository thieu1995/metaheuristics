from models.multiple_solution.swarm_based.PFA import BasePFA, OPFA, LPFA, IPFA
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 10,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
pfa_paras = {
    "epoch": 1000,
    "pop_size": 50
}

## Run model
md = IPFA(root_algo_paras=root_paras, pfa_paras=pfa_paras)
md._train__()

