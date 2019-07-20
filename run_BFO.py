from models.multiple_solution.swarm_based.BFO import BaseBFO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
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

