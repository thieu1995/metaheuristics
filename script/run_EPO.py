from models.multiple_solution.bio_inspired.EPO import BaseEPO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
epo_paras = {
    "epoch": 1000,
    "pop_size": 80
}

## Run model
md = BaseEPO(root_algo_paras=root_paras, epo_paras=epo_paras)
md._train__()

