from models.multiple_solution.swarm_based.SFO import BaseSFO, ImprovedSFO, LevySFO
from utils.FunctionUtil import *

## Setting parameters
root_paras = {
    "problem_size": 100,
    "domain_range": [-1, 1],
    "print_train": True,
    "objective_func": C21
}

# sfo_paras = {
#     "epoch": 500,
#     "pop_size": 100,             # SailFish pop size
#     "pp": 0.2,                  # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
#     "A": 4,                     # A = 4, 6,...
#     "epxilon": 0.001           # = 0.0001, 0.001
# }
#
# md = LevySFO(root_algo_paras=root_paras, sfo_paras=sfo_paras)
# md._train__()



isfo_paras = {
    "epoch": 500,
    "pop_size": 100,             # SailFish pop size
    "pp": 0.1                  # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
}

md = ImprovedSFO(root_algo_paras=root_paras, isfo_paras=isfo_paras)
md._train__()



## Idea: How to remove 2 parameters A and epxilon but the Energy equation (EE) still working?
##      ===> Change the EE based on what we have (current epoch)
## Idea: Apply the fact happended when Sailfish attack Sardines (big ball)
##   The original authors proposed that if AttackPower < 0.5, some sardines will change location, but the fact all sardines will change their location
##   (Watch videos: https://www.youtube.com/watch?v=pRyFGSTaQ_Y, https://www.youtube.com/watch?v=G8x3xpWwMps, )
##   So, no need condition, AP < 0.5 ==> But we still need to take AP into account
##   ===> Based on Opposition-based learning and Energe equation I come up with
##      X_s_new = Ub + Lb - (X_sf_gbest + AP) / 2 + rand(0,1) * ( (X_sf_gbest + AP)/2 - X_s_old )

## Another comment on original paper, their "pp" parameter should be more lower.
## Because in fact, sardines population >> sailfish population (watch video above again)
##      About: pp = 0.001 or 0.0001
## (Link: https://www.newscientist.com/article/2111013-sword-slashing-sailfish-hint-at-origins-of-cooperative-hunting/)


