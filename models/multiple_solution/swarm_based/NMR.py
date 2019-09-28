import numpy as np
from math import gamma
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseNMR(RootAlgo):
    """
    The naked mole-rat algorithm
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, nmr_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = nmr_paras["epoch"]
        self.pop_size = nmr_paras["pop_size"]
        self.size_b = int(self.pop_size / 5)
        self.size_w = self.pop_size - self.size_b
        self.bp = nmr_paras["bp"]       # breeding probability (0.5)

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        g_best = deepcopy(pop[self.ID_MIN_PROBLEM])

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                if i < self.size_b:                     # breeding operators
                    if np.random.uniform() < self.bp:
                        alpha = np.random.uniform()
                        temp = (1 - alpha) * pop[i][self.ID_POS] + alpha * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                else:                                   # working operators
                    t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
                    temp = pop[i][self.ID_POS] + np.random.uniform() * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])

                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            current_best = deepcopy(pop[self.ID_MIN_PROBLEM])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_FIT], self.loss_train


class LevyNMR(RootAlgo):
    """
    The naked mole-rat algorithm
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, nmr_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = nmr_paras["epoch"]
        self.pop_size = nmr_paras["pop_size"]
        self.size_b = int(self.pop_size / 5)
        self.size_w = self.pop_size - self.size_b
        self.bp = nmr_paras["bp"]  # breeding probability (0.5)

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy)
        v = np.random.normal(0, sigma_v)
        s = muy / np.power(np.abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__(minmax=self.ID_MIN_PROBLEM)
        LB =0.001 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        return levy

        # x_new = solution[0] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        # return x_new

    def _crossover__(self, pop, g_best):
        start_point = np.random.randint(0, self.problem_size / 2)
        id1 = start_point
        id2 = int(start_point + self.problem_size / 3)
        id3 = int(self.problem_size)

        partner = pop[np.random.randint(0, self.pop_size)][self.ID_POS]
        new_temp = np.zeros(self.problem_size)
        new_temp[0:id1] = g_best[self.ID_POS][0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best[self.ID_POS][id2:id3]
        return new_temp

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        g_best = deepcopy(pop[self.ID_MIN_PROBLEM])

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                if i < self.size_b:  # breeding operators
                    if np.random.uniform() < self.bp:
                        alpha = np.random.uniform()
                        temp = pop[i][self.ID_POS] + alpha * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        temp = self._crossover__(pop, g_best)
                else:  # working operators
                    if np.random.uniform() < 0.5:
                        t1, t2 = np.random.choice(range(self.size_b, self.pop_size), 2, replace=False)
                        temp = pop[i][self.ID_POS] + np.random.uniform() * (pop[t1][self.ID_POS] - pop[t2][self.ID_POS])
                    else:
                        temp = self._levy_flight__(epoch, pop[i], g_best)

                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            current_best = deepcopy(pop[self.ID_MIN_PROBLEM])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_FIT], self.loss_train
