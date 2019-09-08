from models.multiple_solution.physics_based.TWO import BaseTWO, OTWO, LevyTWO, ITWO
from utils.FunctionUtil import *

root_paras = {
    "problem_size":30,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C28
}
two_paras = {
    "epoch": 1000,
    "pop_size": 100,
}

## Run model
md = ITWO(root_algo_paras=root_paras, two_paras=two_paras)
a, b, c = md._train__()
print(b)
print(a)
print(c)