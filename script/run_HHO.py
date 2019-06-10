from models.multiple_solution.swarm_based.HHO import BaseHHO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 10,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
hho_paras = {
    "epoch": 1000,
    "pop_size": 200
}

## Run model
md = BaseHHO(root_algo_paras=root_paras, hho_paras=hho_paras)
md._train__()

