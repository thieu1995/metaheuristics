import numpy as np
#### Taken from here:
# https://www.robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
# https://arxiv.org/pdf/1003.1409.pdf
# https://sci-hub.tw/10.1080/00207160108805080

## Unimodal benchmark functions
def whale_f1(solution=None, problem_size=None):
    return np.sum(np.power(solution, 2))

def whale_f2(solution=None, problem_size=None):
    return np.sum(np.abs(solution)) + np.prod(np.abs(solution))

def whale_f3(solution=None, problem_size=None):
    return np.sum([ np.power(np.sum([solution[j] for j in range(0, i)]), 2) for i in range(0, problem_size)])

def whale_f5(solution=None, problem_size=None):
    t1 = 0.0
    for i in range(1, problem_size):
        t1 += 100*(solution[i] - solution[i-1]**2)**2 + (solution[i-1] - 1)**2
    return t1

def whale_f6(solution=None, problem_size=None):
    return np.sum( [ np.power( solution[x] + 0.5, 2 ) for x in range(0, problem_size)] )

def whale_f7(solution=None, problem_size=None):
    return np.sum([ i * solution[i]**4 for i in range(problem_size) ]) + np.random.uniform(0, 1)


## Multimodal benchmark functions
def whale_f8(solution=None, problem_size=None):
    t1 = 0.0
    for i in range(problem_size):
        t1 += -solution[i] * np.sin(np.sqrt(np.abs(solution[i])))
    return t1

def whale_f9(solution=None, problem_size=None):
    t1 = 0.0
    for i in range(problem_size):
        t1 += solution[i]**2 - 10*np.cos(2*np.pi*solution[i]) + 10
    return t1

def whale_f10(solution=None, problem_size=None):
    t1 = np.sum(np.power(solution, 2))
    t2 = np.sum([np.cos(2*np.pi*solution[i]) for i in range(problem_size)])
    return -20*np.exp(-0.2*np.sqrt(t1 / problem_size)) - np.exp(t2 / problem_size) + 20 + np.e

def whale_f11(solution=None, problem_size=None):
    t1 = np.sum(np.power(solution, 2))
    t2 = np.prod([ np.cos(solution[i] / np.sqrt(i+1)) for i in range(problem_size) ])
    return t1/4000 - t2 + 1


def square_function(solution=None, problem_size=None):
    return np.sum([solution[i] ** 2 for i in range(0, problem_size)])
