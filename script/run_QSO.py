from models.multiple_solution.human_based.QSO import LevyOppQSO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": whale_f1
}
qso_paras = {
    "epoch": 500,
    "pop_size": 100,
}

## Run model
md = LevyOppQSO(root_algo_paras=root_paras, qso_paras=qso_paras)
md._train__()
