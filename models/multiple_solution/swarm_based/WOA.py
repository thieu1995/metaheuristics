import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseWOA(RootAlgo):
    """
    Standard version of Whale Optimization Algorithm (belongs to Swarm-based Algorithms)
    - In this algorithms: Prey means the best solution
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, woa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = woa_paras["epoch"]
        self.pop_size = woa_paras["pop_size"]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
        # Find prey which is the best solution

        for i in range(self.epoch):
            a = 2 - 2 * i / (self.epoch - 1)            # linearly decreased from 2 to 0

            for j in range(self.pop_size):

                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                p = 0.5
                b = 1
                if (np.random.uniform() < p) :
                    if np.abs(A) < 1:
                        D = np.abs(C * gbest[self.ID_POS] - pop[j][self.ID_POS] )
                        new_position = gbest[0] - A * D
                    else :
                        #x_rand = pop[np.random.randint(self.pop_size)] # chon ra 1 thang random
                        x_rand = self._create_solution__()
                        D = np.abs(C * x_rand[self.ID_POS] - pop[j][self.ID_POS])
                        new_position = (x_rand[self.ID_POS] - A * D)
                else:
                    D1 = np.abs(gbest[0] - pop[j][0])
                    new_position = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest[self.ID_POS]

                new_position[new_position < self.domain_range[0]] = self.domain_range[0]
                new_position[new_position > self.domain_range[1]] = self.domain_range[1]
                fit = self._fitness_model__(new_position)
                pop[j] = [new_position, fit]

            current_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Best fit so far = {}".format(i + 1, gbest[self.ID_FIT]))

        return gbest[self.ID_POS], self.loss_train



class BaoWOA(RootAlgo):
    """
    Code of Bao
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, woa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = woa_paras["epoch"]
        self.pop_size = woa_paras["pop_size"]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)   # Find prey which is the best solution

        for i in range(self.epoch):
            a = 2 - 2 * i / (self.epoch - 1)            # linearly decreased from 2 to 0

            for j in range(self.pop_size):

                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                p = np.random.rand()
                b = 1

                if (p < 0.5) :
                    if np.abs(A) < 1:
                        D = np.abs(C * gbest[self.ID_POS] - pop[j][self.ID_POS] )
                        new_position = gbest[0] - A * D
                    else :
                        x_rand = pop[np.random.randint(self.pop_size)] # chon ra 1 thang random
                        D = np.abs(C * x_rand[self.ID_POS] - pop[j][self.ID_POS])
                        new_position = (x_rand[self.ID_POS] - A * D)
                else:
                    D1 = np.abs(gbest[0] - pop[j][0])
                    new_position = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest[self.ID_POS]
                new_position = self._amend_solution_and_return__(new_position)
                fit = self._fitness_model__(new_position)
                pop[j] = [new_position, fit]

            current_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Fit = {}".format(i + 1, gbest[self.ID_FIT]))
        #print(gbest[self.ID_POS])
        return gbest[self.ID_FIT], self.loss_train
