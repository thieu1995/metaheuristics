from models.multiple_solution.physics_based.NRO import BaseNRO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 10,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
nro_paras = {
    "epoch": 1000,
    "pop_size": 200
}

## Run model
md = BaseNRO(root_algo_paras=root_paras, nro_paras=nro_paras)
md._train__()

