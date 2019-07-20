from models.multiple_solution.swarm_based.PSO import BasePSO
from utils.FunctionUtil import square_function

## Setting parameters
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
pso_paras = {
    "epoch": 100,
    "pop_size": 250,
    "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird
    "c_minmax": [1.2, 1.2]      # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Effecting of  local va global
}

## Run model
md = BasePSO(root_algo_paras=root_paras, pso_paras=pso_paras)
md._train__()

