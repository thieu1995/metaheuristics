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


#### New implemnt taken from here
## Harris Hawks Optimization: Algorithm and Applications

## Unimodal benchmark functions
def hho_f1(solution=None, problem_size=None):
    return np.sum(solution**2)

def hho_f2(solution=None, problem_size=None):
    return np.sum(np.abs(solution)) + np.prod(np.abs(solution))

def hho_f3(solution=None, problem_size=None):
    return np.sum([ (np.sum(solution[0:i]))**2 for i in range(1, problem_size+1)])

def hho_f4(solution=None, problem_size=None):
    return np.max(np.abs(solution))

def hho_f5(solution=None, problem_size=None):
    return np.sum(100*(solution[1:problem_size] - (solution[0:problem_size-1]))**2 + (solution[0:problem_size-1]-1)**2)

def hho_f6(solution=None, problem_size=None):
    return np.sum( (np.abs(solution + 0.5))**2 )

def hho_f7(solution=None, problem_size=None):
    w = [i + 1 for i in range(problem_size)]
    return np.sum( w*(solution**4) ) + np.random.uniform(0, 1)

## Multimodal benchmark functions
def hho_f8(solution=None, problem_size=None):
    return np.sum( -solution * np.sin( np.sqrt(np.abs(solution)) ) )

def hho_f9(solution=None, problem_size=None):
    return np.sum(np.abs( solution**2 - 10*np.cos(2*np.pi*solution) + 10 ))

def hho_f10(solution=None, problem_size=None):
    #return -20 * np.exp(-0.2 * np.sqrt(1.0 / problem_size * np.sum(solution ** 2))) - np.exp(1.0 / problem_size * np.sum(2 * np.pi * solution) + 20 + np.e)
    return -20*np.exp(-0.2*np.sqrt(np.sum(solution**2)/problem_size))-np.exp(np.sum(np.cos(2*np.pi*solution))/problem_size)+20+np.exp(1)

def hho_f11(solution=None, problem_size=None):
    # return 1.0 / 4000 * np.sum(solution ** 2) - np.prod(np.cos(solution / np.sqrt(solution))) + 1
    w=[i for i in range(problem_size)]
    w=[i+1 for i in w]
    return np.sum(solution**2)/4000-np.prod(np.cos(solution/np.sqrt(w)))+1


def hho_ufunction(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))

def hho_f12(solution=None, problem_size=None):
    return (np.pi / problem_size) * (10 * ((np.sin(np.pi * (1 + (solution[0] + 1) / 4))) ** 2) + np.sum(
        (((solution[1:problem_size - 1] + 1) / 4) ** 2) * (1 + 10 * ((np.sin(np.pi * (1 + (solution[1:problem_size - 1]
        + 1) / 4)))) ** 2)) + ( (solution[problem_size - 1] + 1) / 4) ** 2) + np.sum(hho_ufunction(solution, 10, 100, 4))

def hho_f13(solution=None, problem_size=None):
    return 0.1*((np.sin(3*np.pi*solution[0]))**2+np.sum((solution[0:problem_size-2]-1)**2*(1+(np.sin(
        3*np.pi*solution[1:problem_size-1]))**2)) + ((solution[problem_size-1]-1)**2)*(1+(np.sin(2*np.pi*
        solution[problem_size-1]))**2)) + np.sum(hho_ufunction(solution,5,100,4))


## Fixed-dimension multimodal benchmark functions

def hho_f14(solution=None, problem_size=None):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
          [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    v = np.matrix(solution)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = np.sum((np.power(H, 6)))
    w = [i for i in range(25)]
    for i in range(0, 24):
        w[i] = i + 1
    return ((1.0 / 500) + np.sum(1. / (w + bS))) ** (-1)

def hho_f15(solution=None, problem_size=None):
    aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
    bK = 1.0 / np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - ((solution[0] * (bK ** 2 + solution[1] * bK)) / (bK ** 2 + solution[2] * bK + solution[3]))) ** 2)

def hho_f16(solution=None, problem_size=None):
    return 4 * (solution[0] ** 2) - 2.1 * (solution[0] ** 4) + (solution[0] ** 6) / 3 + solution[0] * solution[1] - 4 *\
           (solution[1] ** 2) + 4 * (solution[1] ** 4)

def hho_f17(solution=None, problem_size=None):
    return (solution[1] - (solution[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * solution[0] - 6) ** 2 + 10 * (
                1 - 1 / (8 * np.pi)) * np.cos(solution[0]) + 10

def hho_f18(solution=None, problem_size=None):
    return (1 + (solution[0] + solution[1] + 1) ** 2 * (19 - 14 * solution[0] + 3 * (solution[0] ** 2) - 14 * solution[1]
        + 6 * solution[0] * solution[1] + 3 * solution[1] ** 2)) * (30 + (2 * solution[0] - 3 * solution[1]) ** 2 * (
        18 - 32 * solution[0] + 12 * (solution[0] ** 2) + 48 * solution[1] - 36 * solution[0] * solution[1] + 27 *
        (solution[1] ** 2)))


# map the inputs to the function blocks
def hho_f19(solution=None, problem_size=None):
    aH = np.array([[3, 10, 30], [.1, 10, 35], [3, 10, 30], [.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[.3689, .117, .2673], [.4699, .4387, .747], [.1091, .8732, .5547], [.03815, .5743, .8828]])
    output = 0
    for i in range(0, 4):
        output = output - cH[i] * np.exp(-(np.sum(aH[i, :] * ((solution - pH[i, :]) ** 2))))
    return output

def hho_f20(solution=None, problem_size=None):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [.05, 10, 17, .1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, .05, 10, .1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[.1312, .1696, .5569, .0124, .8283, .5886], [.2329, .4135, .8307, .3736, .1004, .9991],
          [.2348, .1415, .3522, .2883, .3047, .6650], [.4047, .8828, .8732, .5743, .1091, .0381]])
    output = 0
    for i in range(0, 4):
        output = output - cH[i] * np.exp(-(np.sum(aH[i, :] * ((solution - pH[i, :]) ** 2))))
    return output

def hho_f21(solution=None, problem_size=None):
    aSH = [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3],
           [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(0, 4):
        v = np.matrix(solution - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o

def hho_f22(solution=None, problem_size=None):
    aSH = [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3],
           [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(0, 6):
        v = np.matrix(solution - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def hho_f23(solution=None, problem_size=None):
    aSH = [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3],
           [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(0, 9):
        v = np.matrix(solution - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


