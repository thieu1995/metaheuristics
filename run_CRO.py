from models.multiple_solution.evolutionary_based.CRO import BaseCRO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}

cro_paras = {
    "epoch": 500,
    "pop_size": 100,
    "G": [0.02, 0.2],
    "GCR": 0.1,
    "po": 0.4,
    "Fb": 0.9,
    "Fa": 0.1,
    "Fd": 0.1,
    "Pd": 0.1,
    "k": 3
}

## Run model
md = BaseCRO(root_algo_paras=root_paras, cro_paras=cro_paras)
md._train__()

