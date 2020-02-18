from models.multiple_solution.swarm_based.GSO import GSOWOA
from utils.FunctionUtil import *
import json
import os

function_list = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18, C19, C20, C21, C22,
                 C23, C24, C25, C26, C27, C28, C29, C30]

# function_list = [islo_uni_F1]
dimensions = [50]
GSO_paras = {
        "epoch": [50, 10, 300],
        "pop_size": 150,
        "c_minmax": [2.05, 2.05],
        "w_minmax": [0.4, 0.9],
        "num_subswarm": 15
    }
model_name = "GSOWOA"
n_times = 2
epoch = GSO_paras["epoch"][0]
results = []
dir_path = os.path.join(os.path.dirname(__file__), "results/", model_name)

for dim in dimensions:
    for func in function_list:
        ## Setting parameters`
        root_paras = {
            "problem_size": dim,
            "domain_range": [-100, 100],
            "print_train": False,
            "objective_func": func
        }

        is_done = False
        if os.path.exists(dir_path):
            with open(os.path.join(dir_path, model_name + ".json"), 'rb') as fp:
                existed_results = json.load(fp)
                fp.close()
            for result in existed_results:
                if result['dimension'] == dim and result['function_name'] == func.__name__:
                    is_done = True

        if is_done:
            print("Results already exists with dimension {} and function {}".format(dim, func.__name__))
            continue

        ## Run model and save results
        function_name = func.__name__
        print("starting model {} with function {} running in {} dimension".format(model_name, function_name, dim))
        statistical_history_train_losses = np.zeros((n_times, epoch))
        statistical_final_optimal_values = np.zeros(n_times)

        for i in range(n_times):
            md = GSOWOA(root_algo_paras=root_paras, gso_paras=GSO_paras)
            gbest, train_loss = md._train__()
            # print(statistical_history_train_losses.shape, np.asarray(train_loss).shape)
            statistical_history_train_losses[i] += np.asarray(train_loss).reshape((epoch,))
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

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            with open(os.path.join(dir_path, model_name+".json"), 'w') as fp:
                json.dump([result], fp)
        else:
            results = json.load(open(os.path.join(dir_path, model_name+'.json'), 'rb'))
            results.append(result)
            with open(os.path.join(dir_path, model_name+'.json'), 'w') as fp:
                json.dump(results, fp)
                fp.close()

        print("***************************************************")


## Setting parameters`
# root_paras = {
#     "problem_size": 100,
#     "domain_range": [-100, 100],
#     "print_train": True,
#     "objective_func": islo_compos_F24
# }
# woa_paras = {
#     "epoch": 500,
#     "pop_size": 100
# }
#
# ## Run model
# md = ISLO(root_algo_paras=root_paras, woa_paras=woa_paras)
# md._train__()
