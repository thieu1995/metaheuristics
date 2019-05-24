from models.multiple_solution.physics_based.TWO import BaseTWO, OTWO, LevyTWO
from utils.FunctionUtil import *
## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-10, 10],
    "print_train": True,
    "objective_func": whale_f11
}
two_paras = {
    "epoch": 500,
    "pop_size": 50,
}

## Run model
md = LevyTWO(root_algo_paras=root_paras, two_paras=two_paras)
md._train__()

