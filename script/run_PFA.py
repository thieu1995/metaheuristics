from models.multiple_solution.swarm_based.PFA import BasePFA
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
pfa_paras = {
    "epoch": 500,
    "pop_size": 100
}

## Run model
md = BasePFA(root_algo_paras=root_paras, pfa_paras=pfa_paras)
md._train__()

