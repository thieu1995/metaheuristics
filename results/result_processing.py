import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_names = ['GA', 'PSO', 'SLnO', 'WOA', 'ISLO', 'TWO', 'QSO']
keys = ['dimension', 'function_name', 'mean_history_train_loss', 'mean_final_optimal_value', 'std_final_optimal_value']
function_name_list = ['islo_uni_F1', 'islo_uni_F2', 'islo_uni_F3', 'islo_uni_F4', 'islo_uni_F5', 'islo_uni_F6',
                      'islo_uni_F8', 'islo_multi_F9', 'islo_multi_F10', 'islo_multi_F11', 'islo_multi_F12',
                      'islo_multi_F13', 'islo_multi_F14', 'islo_multi_F15', 'islo_multi_F16', 'islo_hybrid_F17',
                      'islo_hybrid_F18', 'islo_hybrid_F19', 'islo_hybrid_F20', 'islo_hybrid_F21', 'islo_hybrid_F22',
                      'islo_hybrid_F23', 'islo_compos_F24', 'islo_compos_F25', 'islo_compos_F26', 'islo_compos_F27',
                      'islo_compos_F28', 'islo_compos_F29', 'islo_compos_F30']


def get_results_table(model_names):
    all_algo_results = []
    for model_name in model_names:
        results = []
        if model_name != 'ISLO':
            file_paths = [model_name+'/'+model_name+'.json', model_name+'/'+model_name+'_2.json']
            for file_path in file_paths:
                # print(file_path)
                with open(file_path, 'r') as f:
                    results_i = json.load(f)
                    results_i_pd = pd.DataFrame(results_i)
                    # print(results_i_pd.shape[0])
                    results.append(results_i_pd)
                    f.close()
            results = pd.concat(results)

            results_50 = results.loc[results['dimension'] == 50]
            results_all_50 = results_50.assign(algorithm=pd.Series([model_name]*results_50.shape[0]))
            all_algo_results.append(results_all_50)
        else:
            file_path = model_name+'/'+model_name+'_final.json'
            # print(file_path)
            with open(file_path, 'r') as f:
                results_i = json.load(f)
                f.close()
            results = pd.DataFrame(results_i)
            results_50 = results.loc[results['dimension'] == 50]
            # print(results_50.shape[0])
            results_all_50 = results_50.assign(algorithm=pd.Series([model_name] * results_50.shape[0]))
            all_algo_results.append(results_all_50)
    all_algo_results = pd.concat(all_algo_results)
    all_algo_results = all_algo_results[['algorithm', 'function_name',
                                         'mean_final_optimal_value', 'std_final_optimal_value']]
    all_algo_results = all_algo_results.groupby(['function_name'])
    final_table = []
    for function_name, row in all_algo_results:
        row = row[['algorithm', 'mean_final_optimal_value', 'std_final_optimal_value']]
        row['rank'] = 1
        row = row.set_index('algorithm').T
        row['function_name'] = function_name
        final_table.append(row)

    final_table = pd.concat(final_table, axis=0)
    final_table.to_csv('function_results.csv', encoding='utf-8', index=True, float_format='%.2f')


for function_name in function_name_list:
    print(function_name)
    function_mean_losses = []
    for model_name in model_names:
        results = []
        if model_name != 'ISLO':
            file_paths = [model_name + '/' + model_name + '.json', model_name + '/' + model_name + '_2.json']
            for file_path in file_paths:
                # print(file_path)
                with open(file_path, 'r') as f:
                    results_i = json.load(f)
                    results += results_i
                    f.close()
        else:
            file_path = model_name + '/' + model_name + '_final.json'
            # print(file_path)
            with open(file_path, 'r') as f:
                results_i = json.load(f)
                results += results_i
                f.close()

        for result in results:
            if result['function_name'] == function_name and result['dimension'] == 50:
                function_mean_losses.append(result['mean_history_train_loss'])

    for i in range(len(function_mean_losses)):
        loss = np.log10(np.asarray(function_mean_losses[i]))
        # loss = function_mean_losses[i]
        plt.plot(loss, label=model_names[i])
    plt.xlabel('Iteration')
    plt.ylabel('Best fitness (log)')
    plt.title(function_name.split('_')[-1])
    plt.legend()
    plt.savefig('./figures/log/'+function_name+'.png')
    plt.close()
