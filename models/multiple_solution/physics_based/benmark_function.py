import numpy as np
class benmark_function:
    def high_conditioned_elliptic(self, x):
        res = 0
        D = len(x)
        for i in range (D):
            res += np.power(10,6*(i)/(D-1))*np.square(x[i])
        return  res
    def f2_whale(self, x):
        return lb, ub, np.sum(np.abs(x)) + np.prod(np.abs(x))
    def f5_whale(self, x):
        res = 0
        D = len(x)
        for i in range(D-1):
            res += 100*np.square((x[i+1] - np.square(x[i]))) + np.square(x[i] -1)
        return res
    def f3_whale(self, x):
        D = len(x)
        res = 0
        for i in range(D):
            for j in range(i):
                temp = np.square(x[j])
            res += temp
        return lb, ub, res
    def f4_whale(self, x):
        return np.max(np.abs(x))
    def f6_whale(self, x):
        return np.sum(np.square(x+0.5))
b  = benmark_function()
f = b.high_conditioned_elliptic
print(f([1,2,3]))