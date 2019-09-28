from models.multiple_solution.swarm_based.BOA import BaseBOA, OriginalBOA, AdaptiveBOA
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
boa_paras = {
    "epoch": 500,
    "pop_size": 100,
    "c": 0.01,
    "p": 0.8,
    "alpha": [0.1, 0.3]
}

## Run model
md = AdaptiveBOA(root_algo_paras=root_paras, boa_paras=boa_paras)
md._train__()

