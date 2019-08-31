from models.multiple_solution.human_based.QSO import LevyOppQSO, BaseQSO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": islo_compos_F24
}
qso_paras = {
    "epoch": 500,
    "pop_size": 100,
}

## Run model
md = BaseQSO(root_algo_paras=root_paras, qso_paras=qso_paras)
md._train__()
