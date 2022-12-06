"""FHO - Fire Hawk Optimizer Algorithm

Algorithm based on the publication : https://link.springer.com/article/10.1007/s10462-022-10173-w
Original Authors: Mahdi Azizi, Siamak Talatahari & Amir H. Gandomi
Original Python file developped by : Adel Remadi and Hamza Lamsaoub  (Students at CentraleSupélec)
"""

from models.multiple_solution.swarm_based.FHO import Developed_FHO
from models.multiple_solution.swarm_based.FHO import Original_FHO
from utils.FunctionUtil import *
import numpy as np

## Setting parameters  /// Unused in our model...
root_paras = {
    "problem_size": 100,
    "domain_range": [-100, 100],
    "print_train": True,
    "objective_func": C30
}

## Examples of cost functions.
sphere=lambda x : np.linalg.norm(x)**2
exponential = lambda x: -np.exp(-0.5*np.sum(x**2))
def ackley(x):
    sumsquares= np.sum(x**2)
    sumcos = np.sum(np.cos(2 * np.pi * x))
    return -20.0 * np.exp(-0.2 * np.sqrt(1/len(x) * sumsquares))-np.exp(1/len(x) * sumcos) + np.e + 20
def becker_lago(x):
    return np.sum((np.abs(x)-5)**2)
def bird(X):                               #Only works in 2-D (min_bounds and max_bounds must be of dim=2 for this function)    
    x,y = X
    return np.sin(x)*(np.exp(1-np.cos(y))**2)+np.cos(y)*(np.exp(1-np.sin(x))**2)+(x-y)**2

fho_paras = {
    "min_bounds":np.ones(2)*-10,    #Change the number of dimensions by changing the size of np.ones. (Arrays with varying bounds as [-30,-10,-20] can also be set as input)
    "max_bounds":np.ones(2)*10,     #Idem as above. Important, len(max_bounds) must be equal to len(min_bounds)
    "pop_size":200,                 #How many solution candidates to consider. 200 worked well in our tests.
    "cost_function": ackley,        #A few example are proposed above (constants chosen arbitrarily). Bird only works in 2-D.
    "max_generations": 200000       #How many max generations to perform.              
}

## Run model 
# We made a modification to the original algorithm thinking it converges better (especially in higher dimensions).
dev_opti = Developed_FHO(fho_paras=fho_paras)
ori_opti = Original_FHO(fho_paras=fho_paras)

#Developed FHO
fmin_dev,xmin_dev=dev_opti.minimize_FHO()
dev_opti.plot_costs()

#Original FHO
fmin_dev,xmin_dev=ori_opti.minimize_FHO()
ori_opti.plot_costs()
