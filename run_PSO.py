from models.multiple_solution.swarm_based.PSO import BasePSO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
pso_paras = {
    "epoch": 500,
    "pop_size": 100,
    "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird
    "c_minmax": [1.2, 1.2]      # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Effecting of  local va global
}

## Run model
md = BasePSO(root_algo_paras=root_paras, pso_paras=pso_paras)
md._train__()

