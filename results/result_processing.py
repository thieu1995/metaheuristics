import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_names = ['GSO', 'GSOLWOA', 'GSOWOA', 'HGEW', "LWOA"]
keys = ['dimension', 'function_name', 'mean_history_train_loss', 'mean_final_optimal_value', 'std_final_optimal_value']
function_name_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
                      'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29',
                      'C30']
constant = np.arange(100, 3001, 100)


# def get_results_table(model_names):
#     all_algo_results = []
#     for model_name in model_names:
#         file_path_1 = model_name+'/'+model_name+'_20_100.json'
#         # print(file_path)
#         with open(file_path_1, 'r') as f:
#             results_i = json.load(f)
#             f.close()
#         results = pd.DataFrame(results_i)
#         results_50 = results.loc[results['dimension'] == 100]
#         # print(results_50.shape[0])
#         results_all_50 = results_50.assign(algorithm=pd.Series([model_name] * results_50.shape[0]))
#         all_algo_results.append(results_all_50)
#     all_algo_results = pd.concat(all_algo_results)
#     all_algo_results = all_algo_results[['algorithm', 'function_name',
#                                          'mean_final_optimal_value', 'std_final_optimal_value']]
#     all_algo_results = all_algo_results.groupby(['function_name'])
#     final_table = []
#     for function_name, row in all_algo_results:
#         row = row[['algorithm', 'mean_final_optimal_value', 'std_final_optimal_value']]
#         row['rank'] = 1
#         row = row.set_index('algorithm').T
#         row['function_name'] = function_name
#         final_table.append(row)
#
#     final_table = pd.concat(final_table, axis=0)
#     final_table.to_csv('function_results_100D.csv', encoding='utf-8', index=True, float_format='%.2f')
#
#
# get_results_table(model_names)

for i in range(len(function_name_list)):
    function_name = function_name_list[i]
    print(function_name)
    function_mean_losses = []
    for model_name in model_names:
        print(model_name)
        results = []

        file_path = model_name + '/' + model_name + '_20_100.json'
        # print(file_path):
        with open(file_path, 'r') as f:
            results_i = json.load(f)
            results += results_i
            f.close()

        for result in results:
            if result['function_name'] == function_name and result['dimension'] == 20:
                if model_name == "LWOA":
                    mean_history_train_loss = []
                    for j in range(len(result['mean_history_train_loss'])):
                        if (j+1) % 40 == 0:
                            mean_history_train_loss.append(result['mean_history_train_loss'][j])
                    print(len(mean_history_train_loss))
                    mean_history_train_loss = np.asarray(mean_history_train_loss)
                    mean_history_train_loss[0] = 10e10
                    function_mean_losses.append(mean_history_train_loss - constant[i])
                else:
                    mean_history_train_loss = np.asarray(result['mean_history_train_loss'])
                    mean_history_train_loss[0] = 10e10
                    function_mean_losses.append(mean_history_train_loss - constant[i])

    for x in range(len(function_mean_losses)):
        loss = np.log10(np.asarray(function_mean_losses[x]))
        # loss = function_mean_losses[i]
        plt.plot(loss, label=model_names[x])
    plt.xlabel('Function Evaluation')
    plt.ylabel('log(Best fitness - '+str(constant[i])+")")
    plt.xticks(np.arange(10, 51, 10), ["60000", "120000", "180000", "240000", "300000"])
    plt.title(function_name.split('_')[-1]+" with dimension "+str(20))
    plt.legend()
    plt.savefig('./figures/log/'+function_name+'.png')
    plt.close()
