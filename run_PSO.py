from models.multiple_solution.swarm_based.PSO import BasePSO
from utils.FunctionUtil import *
import json
import os

function_list = [islo_uni_F1]

# function_list = [islo_compos_F30]
dimensions = [100]
pso_paras = {
    "epoch": 1000,
    "pop_size": 300,
    "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird
    "c_minmax": [2.05, 2.05]      # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Effecting of  local va global
}
model_name = "PSO"
n_times = 20
epoch = pso_paras["epoch"]
results = []

for dim in dimensions:
    for func in function_list:
        ## Setting parameters`
        root_paras = {
            "problem_size": dim,
            "domain_range": [-100, 100],
            "print_train": False,
            "objective_func": func
        }

        ## Run model and save results
        function_name = func.__name__
        print("starting model {} with function {} running in {} dimension".format(model_name, function_name, dim))
        statistical_history_train_losses = np.zeros((n_times, epoch))
        statistical_final_optimal_values = np.zeros(n_times)

        for i in range(n_times):
            md = BasePSO(root_algo_paras=root_paras, pso_paras=pso_paras)
            gbest, train_loss = md._train__()
            statistical_history_train_losses[i] += np.asarray(train_loss).reshape((pso_paras["epoch"],))
            statistical_final_optimal_values[i] += train_loss[-1]
            print("{} of 20 times: result {}".format(i, train_loss[-1]))

        mean_history_train_loss = np.mean(statistical_history_train_losses, axis=0)
        mean_final_optimal_value = np.mean(statistical_final_optimal_values)
        std_final_optimal_value = np.std(statistical_final_optimal_values)

        result = {
            "dimension": dim,
            "function_name": function_name,
            "mean_history_train_loss": mean_history_train_loss.tolist(),
            "mean_final_optimal_value": format(mean_final_optimal_value, '.2e'),
            "std_final_optimal_value": format(std_final_optimal_value, '.2e')
        }
        results.append(result)
        print("***************************************************")

path = "results/"+model_name
if not os.path.exists(path):
    os.makedirs(path)
with open(path+'/'+model_name+'_2.json', 'w') as fp:
    json.dump(results, fp)

# ## Setting parameters
# root_paras = {
#     "problem_size": 30,
#     "domain_range": [-1, 1],
#     "print_train": True,
#     "objective_func": islo_compos_F24
# }
# pso_paras = {
#     "epoch": 500,
#     "pop_size": 100,
#     "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird
#     "c_minmax": [1.2, 1.2]      # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Effecting of  local va global
# }
#
# ## Run model
# md = BasePSO(root_algo_paras=root_paras, pso_paras=pso_paras)
# md._train__()

