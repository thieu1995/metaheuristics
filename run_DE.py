from models.multiple_solution.evolutionary_based.DE import BaseDE
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
de_paras = {
    "epoch": 100,
    "pop_size": 250,
    "Wf": 0.1,
    "Cr": 0.9
}

## Run model
md = BaseDE(root_algo_paras=root_paras, de_paras=de_paras)
md._train__()

