from models.multiple_solution.evolutionary_based.DE import BaseDE
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
de_paras = {
    "epoch": 500,
    "pop_size": 100,
    "Wf": 0.8,
    "Cr": 0.9
}

## Run model
md = BaseDE(root_algo_paras=root_paras, de_paras=de_paras)
md._train__()

