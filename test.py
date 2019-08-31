from utils.class_utils import Fun
from utils.FunctionUtil import *
from models.multiple_solution.swarm_based.WOA import BaoWOA
epoch = 100
pop_size = 100
problem_size = 30
list_fun = [
            Fun("whale_f1", whale_f1, [-100, 10], 0),
            Fun("whale_f2", whale_f2, [-100, 100], 0),
            Fun("whale_f3", whale_f3, [-100, 100], 0),
            Fun("whale_f4", whale_f4, [-100, 100], 0),            
]
for fun in list_fun:
    print("run {}".format(fun.name))
    root_paras = {
                "problem_size": problem_size,
                "domain_range": fun.range,
                "print_train": False,
                "objective_func": fun
            }
    algo_paras = {
                    "epoch": epoch,
                    "pop_size": pop_size
                }
    md = BaoWOA(root_paras, algo_paras)
    a,x,z = md._train__()
    print(z)