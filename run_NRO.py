from models.multiple_solution.physics_based.NRO import BaseNRO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
nro_paras = {
    "epoch": 500,
    "pop_size": 100
}

## Run model
md = BaseNRO(root_algo_paras=root_paras, nro_paras=nro_paras)
md._train__()

