from models.multiple_solution.physics_based.TWO import BaseTWO, OTWO, LevyTWO
from utils.FunctionUtil import *
import sys 
sys.path.insert(0,"D:\metaheuristics\models")
## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": whale_f1
}
two_paras = {
    "epoch": 500,
    "pop_size": 100,
}

## Run model
md = LevyTWO(root_algo_paras=root_paras, two_paras=two_paras)
md._train__()

