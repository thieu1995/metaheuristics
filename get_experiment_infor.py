import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.FunctionUtil import cal_mean, cal_std
from utils.class_utils import AlgoInfor
"""
Get all information (best fit of 15 runtimes and losses for the best run) of all algorithms.
Save all information in  overall/algo_dict_info.pkl
"""

algos = ['GA', 'PSO', 'ABFOLS', 'CRO', 'ABC', 'WOA', 'QSO', 'IQSO']
path_loss = './history/loss/'
path_best_fit = './history/best_fit/'
algo_dict = {}
# iterate over all algorithms
for name in algos:
    al = AlgoInfor()
    al.name = name
    if name == 'IQSO':
        al.name = 'nQSV'
        print("true")
    print(al.name)
    # iterate over 30 benmark functions
    for i in range(1, 31):
        function_name = 'F' + str(i)
        name_file = name + "_" + function_name
        loss_file = name_file + '_loss.pkl'
        best_fit_file = name_file + '_best_fit.pkl'
        path_file_loss = path_loss + loss_file
        path_file_best_fit = path_best_fit + best_fit_file

        with open(path_file_loss, 'rb') as f:
            loss = pkl.load(f)
        with open(path_file_best_fit, 'rb') as f:
            best_fit = pkl.load(f)
        if name == 'PSO':
            # PSO returns matrix form of loss and best fit
            loss = np.reshape(np.array(loss), -1)
            best_fit = np.reshape(np.array(best_fit), -1)
        elif name == 'ABFOLS':
            # ABFOLS return inversed value of fitness and matrix form of loss
            best_fit = 1 / np.array(best_fit)
            loss = np.array(loss)[:, 0]

        # cal std, mean , worst, best of 15 run times
        std = cal_std(best_fit, i*100)
        mean = cal_mean(best_fit, i*100)
        worst = max(best_fit)
        best = min(best_fit)
        al.std.append(std)
        al.mean.append(mean)
        al.best.append(best)
        al.worst.append(worst)
        al.loss.append(np.array(loss))
        al.best_fit.append(np.array(best_fit))
    algo_dict[al.name] = al

# save infor as pickle file
with open('./history/overall/algo_dict_info.pkl', 'wb') as f:
    pkl.dump(algo_dict, f, pkl.HIGHEST_PROTOCOL)

with open('./history/overall/algo_dict_info.pkl', 'rb') as f:
    alf = pkl.load(f)

print(alf['nQSV'].name)
