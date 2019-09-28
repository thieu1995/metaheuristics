from models.multiple_solution.physics_based.TWO import BaseTWO, OppoTWO, OTWO, LevyTWO, ITWO
from utils.FunctionUtil import *

root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
two_paras = {
    "epoch": 500,
    "pop_size": 100,
}

## Run model
md = ITWO(root_algo_paras=root_paras, two_paras=two_paras)
md._train__()


# GA
# DE
#
# QSO
#
# PSO
# WOA
# HHO
#
# NRO
# HGSO
# TWO
# ITWO