import numpy as np
from copy import deepcopy
from math import gamma
from models.multiple_solution.root_multiple import RootAlgo

class BaseTWO(RootAlgo):
    ID_POS = "POS"
    ID_FIT = "FIT"
    ID_WEIGHT = "WEIGHT"

    def __init__(self, root_algo_paras=None, two_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch =  two_paras["epoch"]
        self.pop_size = two_paras["pop_size"]

    def _create_solution__(self, minmax=0):
        pos = np.random.uniform(self.domain_range[0], self.domain_range[1], (self.problem_size, 1))
        fit = self._fitness_model__(solution=pos, minmax=minmax)
        weight = 0.0
        return {"POS": pos, "FIT": fit, "WEIGHT": weight}

    def _update_weight__(self, pop=None):
        maxx = max(pop, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        minn = min(pop, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        for i in range(self.pop_size):
            pop[i][self.ID_WEIGHT] = (pop[i][self.ID_FIT] - minn) / (maxx - minn) + 1
        return pop

    def _amend_solution_and_return__(self, solution=None, best_solution=None, epoch=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0] or solution[i] > self.domain_range[1]:
                if np.random.random() <= 0.5:
                    solution[i] = best_solution[i] + np.random.randn()/ epoch * (best_solution[i]- solution[i])
                    if solution[i] < self.domain_range[0] or solution[i] > self.domain_range[1]:
                        solution[i] = np.random.uniform()
                else:
                    if solution[i] < self.domain_range[0]:
                       solution[i] = self.domain_range[0]
                    if solution[i] > self.domain_range[1]:
                       solution[i] = self.domain_range[1]
        return solution

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.05
        pop = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
        pop = self._update_weight__(pop)
        pop_new = deepcopy(pop)
        g_best = max(pop, key=lambda x: x[self.ID_FIT])

        for i in range(self.epoch):
            for j in range(self.pop_size):
                for k in range(self.pop_size):
                    if pop[j][self.ID_WEIGHT] < pop[k][self.ID_WEIGHT]:
                        pulling_force = max(pop[j][self.ID_WEIGHT] * muy_s, pop[k][self.ID_WEIGHT] * muy_s)
                        resultant_force = pulling_force - pop[j][self.ID_WEIGHT] * muy_k
                        g = pop[k][self.ID_POS] - pop[j][self.ID_POS]
                        acceleration = ( resultant_force / (pop[j][self.ID_WEIGHT] * muy_k) ) * g
                        delta_x = 0.5 * np.power(delta_t, 2) * acceleration + np.power(alpha, i+1) * beta * \
                                  (self.domain_range[1] - self.domain_range[0]) * np.random.uniform(self.problem_size)
                        pop_new[j][self.ID_POS] += delta_x
                pop_new[j][self.ID_POS] = self._amend_solution_and_return__(pop_new[j][self.ID_POS], g_best[self.ID_POS], i+1)
                pop_new[j][self.ID_FIT] = self._fitness_model__(pop_new[j][self.ID_POS], self.ID_MAX_PROBLEM)

            for j in range(self.pop_size):
                if pop[j][self.ID_FIT] < pop_new[j][self.ID_FIT]:
                    pop[j] = deepcopy(pop_new[j])
            pop = self._update_weight__(pop)
            pop_new = deepcopy(pop)

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i+1, 1.0 / g_best[self.ID_FIT]))
        print(g_best[self.ID_POS])
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
        pop = [self._create_solution__(minmax=1) for _ in range(self.pop_size)]
        pop = self._update_weight__(pop)
        pop_new = deepcopy(pop)
        g_best = max(pop, key=lambda x: x[self.ID_FIT])

        for i in range(self.epoch):
            for j in range(self.pop_size):
                for k in range(self.pop_size):
                    if pop[j][self.ID_WEIGHT] < pop[k][self.ID_WEIGHT]:
                        pulling_force = max(pop[j][self.ID_WEIGHT] * muy_s, pop[k][self.ID_WEIGHT] * muy_s)
                        resultant_force = pulling_force - pop[j][self.ID_WEIGHT] * muy_k
                        g = pop[k][self.ID_POS] - pop[j][self.ID_POS]
                        acceleration = (resultant_force / (pop[j][self.ID_WEIGHT] * muy_k)) * g
                        delta_x = 0.5 * np.power(delta_t, 2) * acceleration + np.power(alpha, i + 1) * beta * \
                            (self.domain_range[1] - self.domain_range[0]) * np.random.uniform(self.problem_size)
                        pop_new[j][self.ID_POS] += delta_x
                pop_new[j][self.ID_POS] = self._amend_solution_and_return__(pop_new[j][self.ID_POS],g_best[self.ID_POS], i + 1)
                pop_new[j][self.ID_FIT] = self._fitness_model__(pop_new[j][self.ID_POS], self.ID_MAX_PROBLEM)

            for j in range(0, self.pop_size):
                if pop[j][self.ID_FIT] < pop_new[j][self.ID_FIT]:
                    pop[j] = deepcopy(pop_new[j])
                else:
                    C_op = self.domain_range[1] * np.ones((self.problem_size, 1)) + \
                           self.domain_range[0] * np.ones((self.problem_size, 1)) +\
                           -1 * g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - pop[j][self.ID_POS])
                    fit_op = self._fitness_model__(C_op, self.ID_MAX_PROBLEM)
                    if fit_op > pop[j][self.ID_FIT]:
                        pop[j] = {"POS": C_op, "FIT": fit_op, "WEIGHT": 0.0}
            pop = self._update_weight__(pop)
            pop_new = deepcopy(pop)

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, 1.0 / g_best[self.ID_FIT]))
        print(g_best[self.ID_POS])
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
        D = self._create_solution__(minmax=1)
        LB = 0.01 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        return levy

        # x_new = solution[self.ID_POS] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        # return x_new

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1
        pop = [self._create_solution__(minmax=1) for _ in range(self.pop_size)]
        pop = self._update_weight__(pop)
        pop_new = deepcopy(pop)
        g_best = max(pop, key=lambda x: x[self.ID_FIT])

        for i in range(self.epoch):
            for j in range(self.pop_size):

                if np.random.uniform() < 0.5:
                    pop_new[j][self.ID_POS] = self._levy_flight__(i, pop_new[j], g_best)
                else:
                    for k in range(self.pop_size):
                        if pop[j][self.ID_WEIGHT] < pop[k][self.ID_WEIGHT]:
                            pulling_force = max(pop[j][self.ID_WEIGHT] * muy_s, pop[k][self.ID_WEIGHT] * muy_s)
                            resultant_force = pulling_force - pop[j][self.ID_WEIGHT] * muy_k
                            g = pop[k][self.ID_POS] - pop[j][self.ID_POS]
                            acceleration = ( resultant_force / (pop[j][self.ID_WEIGHT] * muy_k) ) * g
                            delta_x = 0.5 * np.power(delta_t, 2) * acceleration + np.power(alpha, i+1) * beta * \
                                      (self.domain_range[1] - self.domain_range[0]) * np.random.uniform(self.problem_size)
                            pop_new[j][self.ID_POS] += delta_x
                    pop_new[j][self.ID_POS] = self._amend_solution_and_return__(pop_new[j][self.ID_POS], g_best[self.ID_POS], i+1)
                pop_new[j][self.ID_FIT] = self._fitness_model__(pop_new[j][self.ID_POS], self.ID_MAX_PROBLEM)

            for j in range(self.pop_size):
                if pop[j][self.ID_FIT] < pop_new[j][self.ID_FIT]:
                    pop[j] = deepcopy(pop_new[j])
            pop = self._update_weight__(pop)
            pop_new = deepcopy(pop)

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MAX_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i+1, 1.0 / g_best[self.ID_FIT]))
        print(g_best[self.ID_POS])
        return g_best[self.ID_POS], self.loss_train