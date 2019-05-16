from models.multiple_solution.swarm_based.CSO import BaseCSO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}

cso_paras = {
    "epoch": 100,
    "pop_size": 250,
    "mixture_ratio": 0.15,      # MR - joining seeking mode with tracing mode
    "smp": 50,              # seeking memory pool, 50 clones                (larger is better, but need more time)
    "spc": True,            # self-position considering
    "cdc": 0.8,             # counts of dimension to change                 (larger is better)
    "srd": 0.15,            # seeking range of the selected dimension      (lower is better, but need more time)
    "c1": 0.4,
    "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird

    "selected_strategy": 0  # 0: roulette wheel, 1: random, 2: best fitness, 3: tournament
}

## Run model
md = BaseCSO(root_algo_paras=root_paras, cso_paras=cso_paras)
md._train__()

