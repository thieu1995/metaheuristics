import numpy as np 
import matplotlib as plt
import pickle as pkl
from models.multiple_solution.swarm_based.WOA import BaoWOA
from models.multiple_solution.swarm_based.PSO import BasePSO
from models.multiple_solution.swarm_based.BFO import ABFOLS
from models.multiple_solution.swarm_based.ABC import BaseABC
from models.multiple_solution.evolutionary_based.CRO import BaseCRO
from models.multiple_solution.evolutionary_based.GA import BaseGA
from models.multiple_solution.human_based.QSO import BaseQSO, LevyOppQSO
from models.multiple_solution.physics_based.TWO import BaseTWO
from utils.FunctionUtil import *
"""
GA
PSO
ABC - Artificial bee colony algorithm 2005
ABFOL - Adaptive Bacterial Foraging Optimization - 2012
CRO - The coral reefs optimization algorithm - 2014
CSO - Crisscross optimization algorithm - 2014
TWO - tug of war - 2016
WOA - 2016
QS0 - 2016
IQSO
"""
run_times = 15
problem_size = 30
epoch = 500
pop_size = 100
algo_dicts = {'WOA': BaoWOA, 'QSO': BaseQSO, 'IQSO': LevyOppQSO}
                # GA': BaseGA}#, ,
                #  'ABFOLS': ABFOLS, 'CRO': BaseCRO, 'TWO': BaseTWO,
fun_list = [C1, C2, C3, C4, C5, C6, C7, C8,
            C9, C10, C11, C12, C13, C14, C15,
            C16, C17, C18, C19, C20, C21, C22, C23,
            C24, C25, C26, C27, C28, C29, C30]

global_min = [100, 200, 300, 400, 500, 600, 700,
              800, 900, 1000, 1100, 1200, 1300,
              1400, 1500, 1600, 1700, 1800, 1900,
              2000, 2100, 2200, 2300, 2400, 2500,
              2600, 2700, 2800, 2900, 3000]
# run each algo 15 time with 30 different benmark functions
res = {}
for name, Algo in algo_dicts.items():
    std_list = []
    mean_list = []
    worst_list = []
    best_list = []
    print("-----------------------------------")
    for i in range(len(fun_list)):
        list_best_fit = []
        gbest_fit = np.inf
        gworst_fit = np.inf
        std = 0
        mean = 0
        best_loss = []
        for time in range(run_times):
            print("name {}, fun {}, time {}/{}".format(name, i, time, run_times))
            root_paras = {
                "problem_size": problem_size,
                "domain_range": [-100, 100],
                "print_train": False,
                "objective_func": fun_list[i]
            }

            if name == 'GA':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size,
                    "pc": 0.95,
                    "pm": 0.025
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'PSO':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size,
                    "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird
                    "c_minmax": [1.2, 1.2]      # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Effecting of  local va global
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'ABC':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size,
                    "couple_bees": [16, 4],               # number of bees which provided for good location and other location
                    "patch_variables": [5.0, 0.985],        # patch_variables = patch_variables * patch_factor (0.985)
                    "sites": [3, 1],                        # 3 bees (employed bees, onlookers and scouts), 1 good partition
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'ABFOLS':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size,
                    "Ci": [0.1, 0.001],         # C_s (start), C_e (end)  -=> step size # step size in BFO
                    "Ped": 0.25,                  # p_eliminate
                    "Ns": 4,                      # swim_length
                    "N_minmax": [3, 40],          # (Dead threshold value, split threshold value) -> N_adapt, N_split
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'CRO':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size,
                    "G": [0.02, 0.2],
                    "GCR": 0.1,
                    "po": 0.4,
                    "Fb": 0.9,
                    "Fa": 0.1,
                    "Fd": 0.1,
                    "Pd": 0.1,
                    "k": 3
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'TWO':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size,
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'WOA':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'QSO':
                algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size,
                }
                md = Algo(root_paras, algo_paras)

            elif name == 'IQSO':
                algo_paras = {
                    "epoch": pop_size,
                    "pop_size": pop_size,
                }
                md = Algo(root_paras, algo_paras)
            _, loss_history, best_fit = md._train__()

            if best_fit < gbest_fit:
                gbest_fit = best_fit
                best_loss = loss_history
            list_best_fit.append(best_fit)

        std = cal_std(list_best_fit, global_min[i])
        mean = cal_mean(list_best_fit, global_min[i])
        best = min(list_best_fit)
        worst = max(list_best_fit)

        std_list.append(std)
        worst_list.append(worst)
        best_list.append(best)
        mean_list.append(mean)

        fname = name + "_F" + str(i+1)
        file_loss = fname + "_loss"
        file_best_fit = fname + "_best_fit"
        path_file_loss = './history/loss/' + file_loss
        path_file_best_fit = './history/best_fit/' + file_best_fit

        with open(path_file_loss + ".csv", 'w') as f_loss:
            for loss in best_loss:
                f_loss.write(str(loss) + '\n')

        with open(path_file_loss + ".pkl", 'wb') as fo_loss:
            pkl.dump(best_loss, fo_loss, pkl.HIGHEST_PROTOCOL)

        with open(path_file_best_fit + ".csv", 'w') as f_fit:
            for fit in list_best_fit:
                f_fit.write(str(fit) + '\n')

        with open(path_file_best_fit + ".pkl", 'wb') as fo_fit:
            pkl.dump(list_best_fit, fo_fit, pkl.HIGHEST_PROTOCOL)

    res[name] = {'std': std_list, 'mean': mean_list, 'worst': worst_list, 'best': best_list}
with open('./history/overall/res_woa_qso_iqso.csv', 'w') as f:
    for k, v in res.items():
        f.write(k + ',' + str(v) + '\n')
with open('./history/overall/res_woa_qso_iqso.pkl', 'wb') as f:
    pkl.dump(res, f, pkl.HIGHEST_PROTOCOL)
