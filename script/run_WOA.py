from models.multiple_solution.swarm_based.WOA import BaseWOA, BaoWOA
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
woa_paras = {
    "epoch": 100,
    "pop_size": 250
}

## Run model
md = BaoWOA(root_algo_paras=root_paras, woa_paras=woa_paras)
md._train__()

