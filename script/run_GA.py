from models.multiple_solution.evolutionary_based.GA import BaseGA
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
ga_paras = {
    "epoch": 100,
    "pop_size": 250,
    "pc": 0.95,
    "pm": 0.025
}

## Run model
md = BaseGA(root_algo_paras=root_paras, ga_paras=ga_paras)
md._train__()

