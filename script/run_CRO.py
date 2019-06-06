from models.multiple_solution.evolutionary_based.CRO import BaseCRO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}

cro_paras = {
    "epoch": 100,
    "pop_size": 250,
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

