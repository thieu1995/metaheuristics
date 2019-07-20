from models.multiple_solution.evolutionary_based.CRO import OCRO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
cro_paras = {
    "epoch": 100,
    "pop_size": 250,
    "G": [0.02, 0.2],
    "GCR": 0.1,
    "po": 0.4,
    "Fb": 0.8,
    "Fa": 0.1,
    "Fd": 0.3,
    "Pd": 0.1,
    "k": 3
}
ocro_paras = {
    "restart_count": 55
}

## Run model
md = OCRO(root_algo_paras=root_paras, cro_paras=cro_paras, ocro_paras=ocro_paras)
md._train__()

