import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.FunctionUtil import cal_mean, cal_std


class AlgoInfor:
    def __init__(self):
        self.name = ""
        self.loss = []      
        self.best_fit = []

algos = ['PSO', 'ABFOLS', 'CRO', 'ABC', 'WOA', 'QSO']
path_loss = './history/loss/'
path_best_fit = './history/best_fit/'
algo_dict = {}
for name in algos:
    al = AlgoInfor()
    al.name = name
    print(name)
    for i in range(1, 31):
        function_name = 'F' + str(i)
        name_file = name + "_" + function_name
        loss_file = name_file + '_loss.pkl'
        best_fit_file = name_file + '_best_fit.pkl'
        path_file_loss = path_loss + loss_file
        path_file_best_fit = path_best_fit + best_fit_file
        # df_loss = pd.read_csv(path_file_loss, header=None, names=['loss'],
                            #   dtype={'loss': np.float64})
        # df_best_fit = pd.read_csv(path_file_best_fit, header=None, names=['best_fit'],
                                #   dtype={'best_fit': np.float64})
        with open(path_file_loss, 'rb') as f:
            loss = pkl.load(f)
        with open(path_file_best_fit, 'rb') as f:
            best_fit = pkl.load(f)
        if name == 'PSO':
            loss = np.reshape(np.array(loss), -1)
            best_fit = np.reshape(np.array(best_fit), -1)
        elif name == 'ABFOLS':
            best_fit = 1 / np.array(best_fit)
            loss = np.array(loss)[:, 0]
        al.loss.append(np.array(loss))
        al.best_fit.append(np.array(best_fit))
    algo_dict[name] = al

with open('./history/overall/algo_dict_info.pkl', 'wb') as f:
    pkl.dump(algo_dict, f, pkl.HIGHEST_PROTOCOL)

with open('./history/overall/algo_dict_info.pkl', 'rb') as f:
    alf = pkl.load(f)

print(alf['PSO'].name)
