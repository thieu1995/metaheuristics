from models.multiple_solution.swarm_based.BFO import BaseBFO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
bfo_paras = {
    "pop_size": 50,
    "Ci": 0.05,         # step_size
    "Ped": 0.25,        # p_eliminate
    "Ns": 4,            # swim_length
    "Ned": 5,           # elim_disp_steps
    "Nre": 2,           # reproduction_steps
    "Nc": 10,           # chem_steps
    "attract_repel": [0.1, 0.2, 0.1, 10]    # [ d_attr, w_attr, h_repel, w_repel ]
}

## Run model
md = BaseBFO(root_algo_paras=root_paras, bfo_paras=bfo_paras)
md._train__()

