import numpy as np 
import matplotlib as plt
import pickle as pkl
from multiprocessing import Pool
from sklearn.model_selection import ParameterGrid

from utils.class_utils import Fun
from utils.FunctionUtil import *

from models.multiple_solution.evolutionary_based.GA import BaseGA
from models.multiple_solution.evolutionary_based.DE import BaseDE
from models.multiple_solution.evolutionary_based.CRO import BaseCRO
from models.multiple_solution.swarm_based.PSO import BasePSO
from models.multiple_solution.swarm_based.WOA import BaoWOA
from models.multiple_solution.swarm_based.HHO import BaseHHO
from models.multiple_solution.swarm_based.ABC import BaseABC
from models.multiple_solution.swarm_based.PFA import BasePFA
from models.multiple_solution.swarm_based.PFA import IPFA
from models.multiple_solution.physics_based.TWO import BaseTWO
from models.multiple_solution.physics_based.NRO import BaseNRO
from models.multiple_solution.human_based.QSO import BaseQSO


algo_list = [ 
                #['GA', BaseGA],
                ['DE', BaseDE],
                ['PSO', BasePSO],
                ['CRO', BaseCRO],
                ['WOA', BaoWOA],
                #['HHO', BaseHHO],
                #['ABC', BaseABC],
                #['TWO', BaseTWO],
                #['NRO', BaseNRO],
                #['QSO', BaseQSO],
                ['PFA', BasePFA],
                #['IPFA', IPFA]
            ]

fun_list = [
            Fun(1, "whale_f1", whale_f1, [-100, 100], 0),
            Fun(2, "whale_f2", whale_f2, [-10, 10], 0),
            Fun(3, "whale_f3", whale_f3, [-100, 100], 0),
            Fun(4, "whale_f5", whale_f5, [-30, 30], 0),            
            Fun(5, "whale_f6", whale_f6, [-100, 100], 0), 
            Fun(6, "whale_f7", whale_f7, [-1.28, 1.28], 0), 
            Fun(7, "whale_f8", whale_f8, [-500, 500], -418.9286 * 5), 
            Fun(8, "whale_f9", whale_f9, [-100, 100], 0), 
            Fun(9, "whale_f10", whale_f10, [-5.12, 5.12], 0), 
            Fun(10, "whale_f11", whale_f11, [-32, 32], 0), 
            Fun(11, "whale_f12", whale_f12, [-50, 50], 0), 
            Fun(12, "whale_f13", whale_f13, [-50, 50], 0), 
            Fun(13, "whale_f14", whale_f14, [-50, 50], 0), 
            Fun(14, "whale_f15", whale_f15, [-50, 50], 0),
            Fun(15, "whale_f16", whale_f16, [-10, 10], 0), 
            Fun(16, "whale_f17", whale_f17, [-5, 5], -78.33236),
            Fun(17, "C17", C17, [-100, 100], 0),
            Fun(18, "C18", C18, [-100, 100], 0),
            Fun(19, "C19", C19, [-100, 100], 0),
            Fun(20, "C20", C20, [-100, 100], 0),
            Fun(21, "C21", C21, [-100, 100], 0),
            Fun(22, "C22", C22, [-100, 100], 0),
            Fun(23, "C23", C23, [-100, 100], 0),
            Fun(24, "C24", C24, [-100, 100], 0),
            Fun(25, "C25", C25, [-100, 100], 0),
            Fun(26, "C26", C26, [-100, 100], 0),
            Fun(27, "C27", C27, [-100, 100], 0),
            Fun(28, "C28", C28, [-100, 100], 0),
            Fun(29, "C29", C29, [-100, 100], 0),
            Fun(30, "C30", C30, [-100, 100], 0),
        ]

run_times = 15
problem_size = 30
epoch = 1000
pop_size = 100

def run(para):
    """
    algo = ['GA', GA]
    fun = Fun(1, 'whale_f11', whale_f11, 0)
    """
    name = para['algo'][0]
    Algo = para['algo'][1]
    fun = para['fun']
    root_paras = {
                "problem_size": problem_size,
                "domain_range": fun.range,
                "print_train": False,
                "objective_func": fun
            }

    list_best_fit = []
    gbest_fit = np.inf
    gworst_fit = np.inf
    std = 0
    mean = 0
    best_loss = []

    
    for time in range(run_times):
        
        
        if name == 'GA':
            algo_paras = {
                "epoch": epoch,
                "pop_size": pop_size,
                "pc": 0.95,
                "pm": 0.025
            }
            md = Algo(root_paras, algo_paras)

        elif name == 'DE':
            algo_paras = {
                "epoch": epoch,
                "pop_size": 100,
                "Wf": 0.8,
                "Cr": 0.9
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
        
        elif name == 'NRO':
            algo_paras = {
                "epoch": epoch,
                "pop_size": pop_size,
            }
            md = Algo(root_paras, algo_paras)
        
        elif name == "HHO":
            algo_paras = {
                "epoch": epoch,
                "pop_size": 100,
            }
            md = Algo(root_paras, algo_paras)

        elif name == "PFA":
            algo_paras = {
                "epoch": epoch,
                "pop_size": 100,
            }
            md = Algo(root_paras, algo_paras)
        
        elif name == "IPFA":
            algo_paras = {
                "epoch": epoch,
                "pop_size": 100,
            }
            md = Algo(root_paras, algo_paras)
        
        best_fit, loss_history = md._train__()
        if best_fit < gbest_fit:
                gbest_fit = best_fit
                best_loss = loss_history
        list_best_fit.append(best_fit)
        if time + 1 == run_times:
            mess = "name {}, fun {}, time {}/{} + DONE run time".format(name, fun.name, time + 1, run_times)
            print(mess)
            f = open('./history/progress.txt', 'a+')
            f.write(mess + "\n")
            if fun.id == 30:
                mess2 = "==============DONE WHOLE {} ==========================".format(name)
                print(mess2)
                f.write(mess2 + "\n")
            f.close()
           
        else:
            print("name {}, fun {}, time {}/{} ".format(name, fun.name, time + 1, run_times))
            
    fname = name + "_" + str(fun.id) 
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

# for fun in list_fun:
#     print("run {}".format(fun.name))
#     root_paras = {
#                 "problem_size": problem_size,
#                 "domain_range": fun.range,
#                 "print_train": False,
#                 "objective_func": fun
#             }
#     algo_paras = {
#                     "epoch": epoch,
#                     "pop_size": pop_size
#                 }
#     md = ba(root_paras, algo_paras)
#     a,x,z = md._train__()
#     print(z)

param_grid = {'algo': algo_list,
              'fun': fun_list
             }
p = Pool(8)
#for algo in algo_list:
#    for fun in fun_list:
#        run({'algo':algo, 'fun': fun})
p.map(run, list(ParameterGrid(param_grid)))
