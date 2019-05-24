import numpy as np
from copy import deepcopy
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
        best_fit = max(pop, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        worst_fit = min(pop, key=lambda x: x[self.ID_FIT])[self.ID_FIT]
        for i in range(self.pop_size):
            pop[i][self.ID_WEIGHT] = (pop[i][self.ID_FIT] - worst_fit) / (best_fit - worst_fit) + 1
        return pop

    def _amend_solution_and_return__(self, pop=None, best_solution=None, epoch=None):
        for i in range(0, self.pop_size):
            for j in range(0, self.problem_size):
                if pop[i][self.ID_POS][j] < self.domain_range[0] or pop[i][self.ID_POS][j] > self.domain_range[1]:
                    if np.random.random() <= 0.5:
                        pop[i][self.ID_POS][j] = best_solution[j] + np.random.randn() / epoch * (best_solution[j] - pop[i][self.ID_POS][j])
                    else:
                        if pop[i][self.ID_POS][j] < self.domain_range[0]:
                            pop[i][self.ID_POS][j] = self.domain_range[0]
                        if pop[i][self.ID_POS][j] > self.domain_range[1]:
                            pop[i][self.ID_POS][j] = self.domain_range[1]
        return pop

    def _train__(self):
        muy_s = 1
        muy_k = 1
        delta_t = 1
        alpha = 0.99
        beta = 0.1
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        pop = self._update_weight__(pop)
        pop_new = deepcopy(pop)
        g_best = max(pop, key=lambda x: x[self.ID_FIT])

        for i in range(self.epoch):
            for j in range(self.pop_size):
                for k in range(self.pop_size):
                    if pop[j][self.ID_WEIGHT] < pop[k][self.ID_WEIGHT]:
                        force = max(pop[j][self.ID_WEIGHT] * muy_s, pop[k][self.ID_WEIGHT] * muy_s)
                        resultant_force = force - pop[j][self.ID_WEIGHT] * muy_k
                        g = pop[k][self.ID_POS] - pop[j][self.ID_POS]
                        acceleration = ( resultant_force / (pop[j][self.ID_WEIGHT] * muy_k) ) * g
                        delta_x = 0.5 * np.power(delta_t, 2) * acceleration + np.power(alpha, i+1) * beta * \
                                  (self.domain_range[1] - self.domain_range[0]) * np.random.uniform(self.problem_size)
                        pop_new[j][self.ID_POS] += delta_x

            pop_new = self._amend_solution_and_return__(pop_new, g_best[self.ID_POS], i+1)
            for j in range(self.pop_size):
                pop_new[j][self.ID_FIT] = self._fitness_model__(pop_new[j][self.ID_POS])
            for j in range(self.pop_size):
                if pop[j][self.ID_FIT] < pop_new[j][self.ID_FIT]:
                    pop[j] = deepcopy(pop_new[j])
            pop = self._update_weight__(pop)
            pop_new = deepcopy(pop)

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], self.loss_train
