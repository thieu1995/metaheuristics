from models.multiple_solution.swarm_based.HHO import BaseHHO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}
hho_paras = {
    "epoch": 500,
    "pop_size": 100
}

## Run model
md = BaseHHO(root_algo_paras=root_paras, hho_paras=hho_paras)
md._train__()


# https://ww2.mathworks.cn/matlabcentral/fileexchange/65577-human-learning-optimization-hlo-algorithm?s_tid=FX_rc3_behav
# http://evo-ml.com/2019/03/02/hho/
# http://www.alimirjalili.com/HHO.html
# https://www.researchgate.net/project/Harris-hawks-optimization-HHO-Algorithm-and-applications
# https://www.mathworks.com/matlabcentral/fileexchange/70577-harris-hawks-optimization-hho-algorithm-and-applications/?s_tid=LandingPageTabfx
# https://github.com/aliasghar68/Harris-hawks-optimization-Algorithm-and-applications-
# https://github.com/7ossam81/EvoloPy
# https://codeocean.com/capsule/5851871/tree/v1