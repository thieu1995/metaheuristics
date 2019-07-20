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


def CEC_1(solution=None, problem_size=None, shift=0):
    """
    Rotated High Conditioned Elliptic Function
    x1 = x2 = ... = xn = o
    f(x*) = 100
    """
    res = 0
    constant = np.power(10, 6)
    dim = len(solution)
    for i in range(dim):
        res += np.power(constant, i/dim) * np.square((solution[i] - shift))
    return res 


def CEC_2(solution=None, problem_size=None, shift=0):
    """
    Bent cigar function 
    f(x*) =  200 
    """
    res = 0 
    constant = np.power(10, 6)
    dim = len(solution)
    res = np.square((solution[0] - shift))
    for i in range(1, dim):
        res += constant * np.square((solution[i] -  shift))
    return res 


def CEC_3(solution=None, problem_size=None, shift=0):
    """
    Discus Function
    f(x*) = 300
    """
    x = solution - shift
    constant = np.power(10, 6)
    dim = len(solution)
    res = constant * np.square(x[0])
    for i in range(1, dim):
        res += np.square(x[i])
    return res 


def CEC_4(solution=None, problem_size=None, shift=0):
    """
    rosenbrock Function
    f(x*) = 400
    """
    x = solution - shift
    constant = np.power(10, 6)
    dim = len(solution)
    res = 0
    for i in range(dim - 1):
        res += 100 * np.square(x[i]**2 -  x[i+1]) + np.square(x[i] - 1)
    return res 


def CEC_5(solution=None, problem_size=None, shift=0):
    """
    Ackley’s Function
    """
    x = solution - shift
    dim = len(solution)
    res = 0
    A = 0
    B = 0 
    A += -0.2 * np.sqrt(np.sum(np.square(x)) /  dim)
    B += np.sum(np.cos(2 * np.pi * x)) / dim
    res = -20 * np.exp(A) - np.exp(B) + 20 + np.e
   # print("res", res)
    return res 


def CEC_6(solution=None, problem_size=None, shift=0):
    """
    Weierstrass Function
    """
    x = solution - shift
    dim = len(solution)
    res = 0
    kmax = 1
    a = 0.5 
    b = 3
    A = 0
    B = 0
    for i in range(dim):
        for k in range(kmax + 1):
            A += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))
    for k in range(kmax + 1):
        B += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
    res = A - dim * B
    return res 


def CEC_7(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    A = np.sum(np.square(x))/4000 
    B = 1
    if isinstance(x, np.ndarray):
        dim = len(x)
        for i in range(dim):
            B *= np.cos(x[i]/np.sqrt(i+1))
    else:
        B = np.cos(x)
    res = A - B + 1  
    return res 


def CEC_8(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    dim = len(x)
    res = np.sum(np.square(x)) - 10 * np.sum(np.cos(2 * np.pi * x)) + 10 * dim
    return res


def g9(z, dim):
        if np.abs(z) <= 500:
            return z * np.sin(np.power(np.abs(z), 1/2))
        elif z > 500 : 
            return (500 - z % 500) * np.sin(np.sqrt(np.abs(500 - z % 500)))\
                   - np.square(z - 500) / (10000 * dim)
        else:
            return (z % 500 - 500) * np.sin(np.sqrt(np.abs(z % 500 - 500)))\
                   - np.square(z + 500) / (10000 * dim)

def CEC_9(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    dim = len(x)
    A = 0
    B = 0
    A = 418.9829 * dim
    z = x + 4.209687462275036e+002
    for i in range(dim):
        B += g9(z[i], dim)
    res = A - B
    return res 


def CEC_10(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    dim = len(x)
    A = 1
    B = 0
    for i in range(dim):
        temp = 1
        for j in range(32):
            temp += i * (np.abs(np.power(2, j + 1) * x[i] 
                    - round(np.power(2, j + 1) * x[i]))) / np.power(2, j)
        A *= np.power(temp, 10 / np.power(dim, 1.2))
    B = 10 / np.square(dim) 
    res = B*A - B
    return res 


def CEC_11(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    dim = len(x)
    A = 0
    B = 0
    A = np.power(np.abs(np.sum(np.square(x)) - dim), 1/4) 
    B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
    res = A + B + 0.5
    return res 


def CEC_12(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    dim = len(x)
    A = 0
    B = 0
    A = np.power(np.abs(np.square(np.sum(np.square(x))) - np.square(np.sum(x))), 1/2)
    B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
    res = A + B + 0.5
    return res 


def CEC_13(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    dim = len(x)
    A = 0
    B = 0
    for i in range(dim):
        res += CEC_7(CEC_4(x[i : (i + 2) % dim], shift=0), shift=0)
    return res


def CEC_14(solution=None, problem_size=None, shift=0):
    x  = solution - shift
    res = 0 
    dim = len(x)
    A = 0
    B = 0
    def g(x, y):
        return 0.5 + (np.square(np.sin(np.sqrt(x * x + y * y))) - 0.5) / \
                      np.square(1 + 0.001 * np.square((x*x + y*y)))
    for i in range(dim):
        res += g(x[i], x[(i+1) % dim])
    return res 


def shift(solution, shift_number):
    return np.array(solution) - shift_number


def rotate(solution, original_x , rotate_rate=1):
    return solution


def C1(solution, problem_size=None, shift_num=1, rate=1):
    x = shift(solution, shift_num)
    return CEC_1(x) + 100 * rate


def C2(solution, prolem_size=None, shift_num=1, rate=1):
    x = shift(solution, shift_num)
    return CEC_2(x) + 200 * rate


def C3(solution, prolem_size=None, shift_num=1, rate=1):
    x = shift(solution, shift_num)
    return CEC_3(x) + 300 * rate


def C4(solution, prolem_size=None, shift_num=2, rate=1):
    x = 2.48/100*shift(solution, shift_num)
    x = rotate(x, solution) + 1
    return CEC_4(x) + 400 * rate


def C5(solution, prolem_size=None, shift_num=1, rate=1):
    x = shift(solution, shift_num)
    x = rotate(x, solution)
    return CEC_5(x) + 500 * rate


def C6(solution, prolem_size=None, shift_num=1, rate=1):
    x = 0.5 / 100 * shift(solution, shift_num)
    return CEC_6(x) + 600 * rate


def C7(solution, prolem_size=None, shift_num=1, rate=1):
    x = 600 / 100 * shift(solution, shift_num)
    return CEC_7(x) + 700 * rate


def C8(solution, prolem_size=None, shift_num=1, rate=1):
    x = 5.12 / 100 * shift(solution, shift_num)
    return CEC_8(x) + 800 * rate


def C9(solution, prolem_size=None, shift_num=1, rate=1):
    x = 5.12 / 100 * shift(solution, shift_num)
    x = rotate(x, solution)
    return CEC_8(x) + 900 * rate


def C10(solution, prolem_size=None, shift_num=1, rate=1):
    x = 1000 / 100 * shift(solution, shift_num)
    return CEC_9(x) + 1000 * rate


def C11(solution, prolem_size=None, shift_num=1, rate=1):
    x = 1000 / 100 * shift(solution, shift_num)
    x = rotate(x, solution)
    return CEC_9(x) + 1100 * rate


def C12(solution, prolem_size=None, shift_num=1, rate=1):
    x = 5 / 100 * shift(solution, shift_num)
    x = rotate(x, solution)
    return CEC_10(x) + 1200 * rate

def C13(solution, prolem_size=None, shift_num=1, rate=1):
    x = 5 / 100 * shift(solution, shift_num)
    x = rotate(x, solution)
    return CEC_11(x) + 1300 * rate


def C14(solution, prolem_size=None, shift_num=1, rate=1):
    x = 5 / 100 * shift(solution, shift_num)
    x = rotate(x, solution)
    return CEC_12(x) + 1400 * rate


def C15(solution, prolem_size=None, shift_num=2, rate=1):
    x = 5 / 100 * shift(solution, shift_num)
    x = rotate(x, solution) + 1
    return CEC_13(x) + 1500 * rate


def C16(solution, prolem_size=None, shift_num=1, rate=1):
    x = 5 / 100 * shift(solution, shift_num)
    x = rotate(x, solution) + 1
    return CEC_14(x) + 1600 * rate


def C17(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.3 * dim)
    n2 = int(0.3 * dim) + n1
    D = np.arange(dim)

    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_9(x[D[ : n1]]) + CEC_8(x[D[n1 : n2]]) + CEC_1(x[D[n2 : ]]) + 1700 * rate


def C18(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.3 * dim)
    n2 = int(0.3 * dim) + n1
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_2(x[D[ : n1]]) + CEC_12(x[D[n1 : n2]]) + CEC_8(x[D[n2 : ]]) + 1800 * rate


def C19(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.2 * dim)
    n2 = int(0.2 * dim) + n1
    n3 = int(0.3 * dim) + n2 
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_7(x[D[ : n1]]) + CEC_6(x[D[n1 : n2]]) + CEC_4(x[D[n2 : n3]]) + CEC_14(x[D[n3 : ]]) + 1900 * rate


def C20(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.2 * dim)
    n2 = int(0.2 * dim) + n1
    n3 = int(0.3 * dim) + n2 
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_12(x[D[ : n1]]) + CEC_3(x[D[n1 : n2]]) + CEC_13(x[D[n2 : n3]]) + CEC_8(x[D[n3 : ]]) + 2000 * rate


def C21(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.1 * dim)
    n2 = int(0.2 * dim) + n1
    n3 = int(0.2 * dim) + n2 
    n4 = int(0.2 * dim) + n3
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_14(x[D[ : n1]]) + CEC_12(x[D[n1 : n2]]) + CEC_4(x[D[n2 : n3]]) + CEC_9(x[D[n3 : n4]]) + CEC_1(x[D[n4 : ]]) + 2100 * rate


def C22(solution, prolem_size=None, shift_num=1, rate=1):
    dim = len(solution)
    n1 = int(0.1 * dim)
    n2 = int(0.2 * dim) + n1
    n3 = int(0.2 * dim) + n2 
    n4 = int(0.2 * dim) + n3
    D = np.arange(dim)
    # np.random.shuffle(D)
    x = shift(solution, shift_num)
    return CEC_10(x[D[ : n1]]) + CEC_11(x[D[n1 : n2]]) + CEC_13(x[D[n2 : n3]]) + CEC_9(x[D[n3 : n4]]) +\
            CEC_5(x[D[n4 : ]]) + 2200 * rate


def C23(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 20, 30, 40, 50]
    lamda = [1, 1.0e-6, 1.0e-26, 1.0e-6, 1.0e-6 ]
    bias = [0, 100, 200, 300, 400]
    fun = [C4, C1, C2, C3, C1]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 2300 * rate


def C24(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [1, 2, 3]
    sigma = [20, 20, 20]
    lamda = [1, 1, 1]
    bias = [0, 100, 200]
    fun = [C10, C9, C14]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
               * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 2400 * rate


def C25(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [1, 2, 3]
    sigma = [10, 30, 50]
    lamda = [0.25, 1, 1.0e-7]
    bias = [0, 100, 200]
    fun = [C11, C9, C1]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
               * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 2500 * rate


def C26(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 10, 10, 10, 10]
    lamda = [0.25, 1.0, 1.0e-7, 2.5, 10.0 ]
    bias = [0, 100, 200, 300, 400]
    fun = [C11, C13, C1, C6, C7]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
               * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 2600 * rate


def C27(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 10, 10, 20, 20]
    lamda = [10, 10, 2.5, 25, 1.0e-6 ]
    bias = [0, 100, 200, 300, 400]
    fun = [C14, C9, C11, C6, C1]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
               * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 2700 * rate


def C28(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [1, 2, 3, 4, 5]
    sigma = [10, 20, 30, 40, 50]
    lamda = [2.5, 10, 2.5, 5.0e-4, 1.0e-6]
    bias = [0, 100, 200, 300, 400]
    fun = [C15, C13, C11, C16, C1]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
               * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 2800 * rate


def C29(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [4, 5, 6]
    sigma = [10, 30, 50]
    lamda = [1, 1, 1]
    bias = [0, 100, 200]
    fun = [C17, C18, C19]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
               * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 2900 * rate


def C30(solution, prolem_size=None, shift_num=1, rate=1):
    shift_arr = [1, 2, 3]
    sigma = [10, 30, 50]
    lamda = [1, 1, 1]
    bias = [0, 100, 200]
    fun = [C20, C21, C22]
    dim = len(solution)
    res = 0
    w = np.zeros(len(shift_arr))
    for i in range(len(shift_arr)):
        x = shift(solution, shift_arr[i])
        w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
               * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
    for i in range(len(shift_arr)):
        res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
    return res + 3000 * rate


def cal_mean(li, global_min):
    return np.mean(np.array(li) - global_min)


def cal_std(li, global_min):
    return np.sqrt(np.mean(np.square(np.array(li) - global_min)))



if __name__ == "__main__":

    x = np.array([1, 1, 1 , 1, 1])
    print(C1(x))    



        
    

    
    




