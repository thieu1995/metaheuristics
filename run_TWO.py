from models.multiple_solution.physics_based.TWO import BaseTWO, OppoTWO, OTWO, LevyTWO, ITWO
from utils.FunctionUtil import *

root_paras = {
    "problem_size": 50,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": islo_compos_F24
}
two_paras = {
    "epoch": 500,
    "pop_size": 100,
}

## Run model
md = BaseTWO(root_algo_paras=root_paras, two_paras=two_paras)
md._train__()
