from models.multiple_solution.human_based.PO import PO
from utils.FunctionUtil import square_function

## Setting parameters
#we don't use, should we? 
"""
root_paras = {
    "problem_size": 30,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": square_function
}
"""



po_params = {
    'fun':lambda x: x[0]**2 + x[1]**2, #the function we want to optimize
    'n':3,                             #the number of contituencies, political parties and party members
    'tmax':100,                        #number of iterations
    'd':2 ,                            #number of dimensions, we don't want this to be a parameter later I think
    'lambdamax' : 2                    #upper limit of the party switching rate            
}


## Run model
po = PO(po_params = po_params)
po._train__()
