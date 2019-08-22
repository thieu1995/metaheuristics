import numpy as np
from copy import deepcopy
from math import gamma
from models.multiple_solution.root_multiple import RootAlgo

class BaseTWO(RootAlgo):
    """
    A novel meta-heuristic algorithm: tug of war optimization
    """
    ID_POS = 0
    ID_FIT = 1
    ID_WEIGHT = 2

    def __init__(self, root_algo_paras=None, two_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = two_paras["epoch"]
        self.pop_size = two_paras["pop_size"]

    def _create_solution__(self, minmax=0):
        solution = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution, minmax=minmax)
        weight = 0.0
        return [solution, fitness, weight]

    def _update_weight__(self, teams):
        best_fitness = max(teams, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        worst_fitness = min(teams, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        for i in range(self.pop_size):
            teams[i][self.ID_WEIGHT] = (teams[i][self.ID_FIT] - worst_fitness)/(best_fitness - worst_fitness) + 1
        return teams

    def _update_fit__(self, teams):
        for i in range(self.pop_size):
            teams[i][self.ID_FIT] = self._fitness_model__(teams[i][self.ID_POS], minmax=self.ID_MAX_PROBLEM)
        return teams

    def _amend_and_return_pop__(self, pop_old, pop_new, g_best, epoch):
        for i in range(self.pop_size):
            for j in range(self.problem_size):
                if pop_new[i][self.ID_POS][j] < self.domain_range[0] or pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                    if np.random.random() <= 0.5:
                        pop_new[i][self.ID_POS][j] = g_best[self.ID_POS][j] + np.random.randn()/(epoch+1)*(g_best[self.ID_POS][j] - pop_new[i][self.ID_POS][j])
                        if pop_new[i][self.ID_POS][j] < self.domain_range[0] or pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                            pop_new[i][self.ID_POS][j] = pop_old[i][self.ID_POS][j]
                    else:
                        if pop_new[i][self.ID_POS][j] < self.domain_range[0]:
                           pop_new[i][self.ID_POS][j] = self.domain_range[0]
                        if pop_new[i][self.ID_POS][j] > self.domain_range[1]:
                           pop_new[i][self.ID_POS][j] = self.domain_range[1]
        return pop_new

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1

        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for j in range( self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                       force = max(pop_old[i][self.ID_WEIGHT]*muy_s, pop_old[j][self.ID_WEIGHT]*muy_s)
                       resultant_force = force - pop_old[i][self.ID_WEIGHT]*muy_k
                       g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                       acceleration = resultant_force*g/(pop_old[i][self.ID_WEIGHT]*muy_k)
                       delta_x = 1/2*acceleration + np.power(alpha,epoch+1)\
                                 *beta*(self.domain_range[1] -  self.domain_range[0])*\
                                np.random.randn(self.problem_size)
                       pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch+1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, 1.0 / g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train, g_best[self.ID_FIT]


class OppoTWO(BaseTWO):
    def __init__(self, root_algo_paras=None, two_paras = None):
        BaseTWO.__init__(self, root_algo_paras, two_paras)

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1

        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                        force = max(pop_old[i][self.ID_WEIGHT] * muy_s, pop_old[j][self.ID_WEIGHT] * muy_s)
                        resultant_force = force - pop_old[i][self.ID_WEIGHT] * muy_k
                        g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                        acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * muy_k)
                        delta_x = 1 / 2 * acceleration + np.power(alpha, epoch + 1) \
                                  * beta * (self.domain_range[1] - self.domain_range[0]) * \
                                  np.random.randn(self.problem_size)
                        pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch + 1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    t1 = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * np.ones(self.problem_size)
                    t2 = -1 * g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - pop_old[i][self.ID_POS])
                    C_op = t1 + t2
                    fit_op = self._fitness_model__(C_op, self.ID_MAX_PROBLEM)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, 1.0 / g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train


class OTWO(BaseTWO):
    def __init__(self, root_algo_paras=None, two_paras = None):
        BaseTWO.__init__(self, root_algo_paras, two_paras)

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1

        pop_temp = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
        pop_oppo = deepcopy(pop_temp)
        for i in range(self.pop_size):
            item_oppo = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * np.ones(self.problem_size) - pop_temp[i][self.ID_POS]
            fit = self._fitness_model__(item_oppo, self.ID_MAX_PROBLEM)
            pop_oppo[i] = [item_oppo, fit, 0.0]
        pop_oppo = pop_temp + pop_oppo
        pop_old = sorted(pop_oppo, key=lambda item: item[self.ID_FIT])[self.pop_size:]
        pop_old = self._update_weight__(pop_old)

        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                        force = max(pop_old[i][self.ID_WEIGHT] * muy_s, pop_old[j][self.ID_WEIGHT] * muy_s)
                        resultant_force = force - pop_old[i][self.ID_WEIGHT] * muy_k
                        g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                        acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * muy_k)
                        delta_x = 1 / 2 * acceleration + np.power(alpha, epoch + 1) \
                                  * beta * (self.domain_range[1] - self.domain_range[0]) * \
                                  np.random.randn(self.problem_size)
                        pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch + 1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    t1 = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * np.ones(self.problem_size)
                    t2 = -1 * g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - pop_old[i][self.ID_POS])
                    C_op = t1 + t2
                    fit_op = self._fitness_model__(C_op, self.ID_MAX_PROBLEM)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, 1.0 / g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train




class LevyTWO(BaseTWO):
    def __init__(self, root_algo_paras=None, two_paras = None):
        BaseTWO.__init__(self, root_algo_paras, two_paras)

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy)
        v = np.random.normal(0, sigma_v)
        s = muy / np.power(np.abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__(minmax=self.ID_MAX_PROBLEM)
        LB = 0.01 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        return levy

        #x_new = solution[0] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        #return x_new

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1

        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                if np.random.uniform() < 0.5:
                    pop_new[i][self.ID_POS] = self._levy_flight__(epoch+1, pop_new[i], g_best)
                else:
                    for j in range( self.pop_size):
                        if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                           force = max(pop_old[i][self.ID_WEIGHT]*muy_s, pop_old[j][self.ID_WEIGHT]*muy_s)
                           resultant_force = force - pop_old[i][self.ID_WEIGHT]*muy_k
                           g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                           acceleration = resultant_force*g/(pop_old[i][self.ID_WEIGHT]*muy_k)
                           delta_x = 1/2*acceleration + np.power(alpha,epoch+1)*beta*\
                                     (self.domain_range[1] -  self.domain_range[0])*np.random.randn(self.problem_size)
                           pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch+1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, 1.0 / g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train



class ITWO(OppoTWO, LevyTWO):
    def __init__(self, root_algo_paras=None, two_paras = None):
        OppoTWO.__init__(self, root_algo_paras, two_paras)

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1

        pop_old = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
        pop_old = self._update_weight__(pop_old)
        g_best = max(pop_old, key=lambda x: x[self.ID_FIT])
        pop_new = deepcopy(pop_old)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                if np.random.uniform() < 0.5:
                    pop_new[i][self.ID_POS] = self._levy_flight__(epoch + 1, pop_new[i], g_best)
                else:
                    for j in range(self.pop_size):
                        if pop_old[i][self.ID_WEIGHT] < pop_old[j][self.ID_WEIGHT]:
                            force = max(pop_old[i][self.ID_WEIGHT] * muy_s, pop_old[j][self.ID_WEIGHT] * muy_s)
                            resultant_force = force - pop_old[i][self.ID_WEIGHT] * muy_k
                            g = pop_old[j][self.ID_POS] - pop_old[i][self.ID_POS]
                            acceleration = resultant_force * g / (pop_old[i][self.ID_WEIGHT] * muy_k)
                            delta_x = 1 / 2 * acceleration + np.power(alpha, epoch + 1) * beta * \
                                      (self.domain_range[1] - self.domain_range[0]) * np.random.randn(self.problem_size)
                            pop_new[i][self.ID_POS] += delta_x

            pop_new = self._amend_and_return_pop__(pop_old, pop_new, g_best, epoch + 1)
            pop_new = self._update_fit__(pop_new)

            for i in range(self.pop_size):
                if pop_old[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
                    pop_old[i] = deepcopy(pop_new[i])
                else:
                    C_op = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * np.ones(self.problem_size) + \
                           -1 * g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - pop_old[i][self.ID_POS])
                    fit_op = self._fitness_model__(C_op, self.ID_MAX_PROBLEM)
                    if fit_op > pop_old[i][self.ID_FIT]:
                        pop_old[i] = [C_op, fit_op, 0.0]
            pop_old = self._update_weight__(pop_old)
            pop_new = deepcopy(pop_old)

            current_best = self._get_global_best__(pop_old, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(1.0 / g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, 1.0 / g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train

