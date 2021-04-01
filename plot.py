import pickle as pkl 
from utils.FunctionUtil import cal_mean, cal_std
with open("./history/overall/res.pkl", 'rb') as f:
    res = pkl.load(f)
with open("./history/loss/GA_F1_loss.pkl", 'rb') as f:
    list_best = pkl.load(f)
# std = cal_std(list_best, 100)
# mean = cal_mean(list_best, 100)
# best = min(list_best)
# print(std)
# print(mean)
# print(best)
print(list_best)
# print(res['GA']['std'][1])
# def 