from models.multiple_solution.human_based.QSO import LevyOppQSO, BaseQSO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C28
}
qso_paras = {
    "epoch": 500,
    "pop_size": 100,
}

## Run model
md = BaseQSO(root_algo_paras=root_paras, qso_paras=qso_paras)
a,b,c = md._train__()
print(c)