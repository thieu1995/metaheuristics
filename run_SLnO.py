from models.multiple_solution.swarm_based.SLnO import SLnO
from utils.FunctionUtil import *
import json
import os

function_list = [whale_f1, whale_f2, whale_f3]
SLnO_paras = {
        "epoch": 10,
        "pop_size": 100
    }
model_name = "SLnO"
n_times = 2
epoch = SLnO_paras["epoch"]
results = []

for func in function_list:
    ## Setting parameters`
    root_paras = {
        "problem_size": 30,
        "domain_range": [-100, 100],
        "print_train": True,
        "objective_func": func
    }

    ## Run model and save results

    function_name = func.__name__
    print("starting model {} with function {}".format(model_name, function_name))
    statistical_history_train_losses = np.zeros((n_times, epoch))
    statistical_final_optimal_values = np.zeros(n_times)

    for i in range(n_times):
        md = SLnO(root_algo_paras=root_paras, woa_paras=SLnO_paras)
        gbest, train_loss = md._train__()
        statistical_history_train_losses[i] += np.asarray(train_loss)
        statistical_final_optimal_values[i] += train_loss[-1]

    mean_history_train_loss = np.mean(statistical_history_train_losses, axis=0)
    mean_final_optimal_value = np.mean(statistical_final_optimal_values)
    std_final_optimal_value = np.std(statistical_final_optimal_values)

    result = {
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
with open(path+'/'+model_name+'.json', 'w') as fp:
    json.dump(results, fp)
