class AlgoInfor:
    def __init__(self):
        self.name = ""
        self.loss = []
        self.best_fit = []
        self.mean = []
        self.std = []
        self.best = []
        self.worst = []

class Fun:
    def __init__(self, id, name, fun, range, fmin):
        self.name = name 
        self.range = range 
        self.fmin = fmin
        self.fun = fun 
        self.id = id
    
    def __call__(self, solution, problem_size):
        return self.fun(solution)
