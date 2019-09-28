import pandas as pd
from models.multiple_solution.swarm_based.ABC import *
from models.multiple_solution.swarm_based.BMO import *
from models.multiple_solution.swarm_based.BOA import *
from models.multiple_solution.swarm_based.EPO import *
from models.multiple_solution.swarm_based.HHO import *
from models.multiple_solution.swarm_based.NMR import *
from models.multiple_solution.swarm_based.PFA import *
from models.multiple_solution.swarm_based.PSO import *
from models.multiple_solution.swarm_based.SFO import *
from models.multiple_solution.swarm_based.SOA import *
from models.multiple_solution.swarm_based.WOA import *
from utils.FunctionUtil import *


## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}


abc_paras = {
    "epoch": 500,
    "pop_size": 100,
    "couple_bees": [16, 4],               # number of bees which provided for good location and other location
    "patch_variables": [5.0, 0.985],        # patch_variables = patch_variables * patch_factor (0.985)
    "sites": [3, 1],                        # 3 bees (employed bees, onlookers and scouts), 1 good partition
}

bmo_paras = {
    "epoch": 500,
    "pop_size": 100,
    "bm_teams": 10
}

boa_paras = {
    "epoch": 500,
    "pop_size": 100,
    "c": 0.01,
    "p": 0.8,
    "alpha": [0.1, 0.3]
}

epo_paras = {
    "epoch": 500,
    "pop_size": 100
}

hho_paras = {
    "epoch": 500,
    "pop_size": 100
}

nmr_paras = {
    "pop_size": 100,
    "epoch": 500,
    "bp": 0.75,      # breeding probability
}

pfa_paras = {
    "epoch": 500,
    "pop_size": 100
}

pso_paras = {
    "epoch": 500,
    "pop_size": 100,
    "w_minmax": [0.4, 0.9],     # [0-1] -> [0.4-0.9]      Weight of bird
    "c_minmax": [1.2, 1.2]      # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Effecting of  local va global
}

isfo_paras = {
    "epoch": 500,
    "pop_size": 100,             # SailFish pop size
    "pp": 0.1                  # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
}

soa_paras = {
    "epoch": 500,
    "pop_size": 100,
}

woa_paras = {
    "epoch": 500,
    "pop_size": 100
}


## Run model
name_model = {
    'BaseABC': BaseABC(root_algo_paras=root_paras, abc_paras=abc_paras),
    'BaseBMO': BaseBMO(root_algo_paras=root_paras, bmo_paras=bmo_paras),
    "AdaptiveBOA": AdaptiveBOA(root_algo_paras=root_paras, boa_paras=boa_paras),
    "BaseEPO": BaseEPO(root_algo_paras=root_paras, epo_paras=epo_paras),
    "BaseHHO": BaseHHO(root_algo_paras=root_paras, hho_paras=hho_paras),
    "LevyNMR": LevyNMR(root_algo_paras=root_paras, nmr_paras=nmr_paras),
    "IPFA": IPFA(root_algo_paras=root_paras, pfa_paras=pfa_paras),
    "BasePSO": BasePSO(root_algo_paras=root_paras, pso_paras=pso_paras),
    "ImprovedSFO": ImprovedSFO(root_algo_paras=root_paras, isfo_paras=isfo_paras),
    "BaseSOA": BaseSOA(root_algo_paras=root_paras, soa_paras=soa_paras),
    "BaoWOA": BaoWOA(root_algo_paras=root_paras, woa_paras=woa_paras)
}

### 1st: way
# list_loss = []
# for name, model in name_model.items():
#     _, loss = model._train__()
#     list_loss.append(loss)
# list_loss = np.asarray(list_loss)
# list_loss = list_loss.T
# np.savetxt("run_test_c30.csv", list_loss, delimiter=",", header=str(name_model.keys()))


### 2nd: way
list_loss = {}
for name, model in name_model.items():
    _, loss = model._train__()
    list_loss[name] = loss
df = pd.DataFrame(list_loss)
df.to_csv('c30_results.csv')        # saving the dataframe
