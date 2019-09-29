import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseBMO(RootAlgo):
    """
    The blue monkey: A new nature inspired metaheuristic optimization algorithm
    """
    ID_POS = 0      # position
    ID_FIT = 1      # fitness
    ID_RAT = 2      # rate
    ID_WEI = 3      # weight

    def __init__(self, root_algo_paras=None, bmo_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = bmo_paras["epoch"]
        self.pop_size = bmo_paras["pop_size"]
        self.bm_teams = bmo_paras["bm_teams"]       # Number of blue monkey teams (5, 10, 20, ...)
        self.bm_size = int(self.pop_size/2)         # Number of all blue monkey
        self.bm_numbers = int(self.bm_size / self.bm_teams)     # Number of blue monkey in each team

    def _create_solution__(self, minmax=0):
        solution = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution, minmax=minmax)
        rate = np.random.uniform(0, 1)
        weight = np.random.uniform(4, 6)
        return [solution, fitness, rate, weight]

    def _create_population__(self):
        t1 = []
        for i in range(self.bm_size):
            t2 = [self._create_solution__(self.ID_MIN_PROBLEM) for _ in range(self.bm_numbers)]
            t1.append(t2)
        t2 = [self._create_solution__(self.ID_MIN_PROBLEM) for _ in range(self.bm_size)]
        return t1, t2


    def _train__(self):
        bm_pop, child_pop = self._create_population__()

        best = []
        for items in bm_pop:
            bt = self._get_global_best__(items, self.ID_FIT, self.ID_MIN_PROBLEM)
            best.append(deepcopy(bt))
        g_best = deepcopy(self._get_global_best__(best, self.ID_FIT, self.ID_MIN_PROBLEM))
        child_best = self._get_global_best__(child_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
        if g_best[self.ID_FIT] > child_best[self.ID_FIT]:
            g_best = deepcopy(child_best)

        for epoch in range(self.epoch):

            for items in bm_pop:
                items = sorted(items, key=lambda temp: temp[self.ID_FIT])
                items[-1] = deepcopy(child_best)

            for items in bm_pop:
                leader = self._get_global_best__(items, self.ID_FIT, self.ID_MIN_PROBLEM)
                for i in range(self.bm_numbers):
                    rate = 0.7 * items[i][self.ID_RAT] + (leader[self.ID_WEI] - items[i][self.ID_WEI]) * \
                              np.random.uniform() * (leader[self.ID_POS] - items[i][self.ID_POS])
                    pos = items[i][self.ID_POS] + np.random.uniform() * rate
                    pos = self._amend_solution_and_return__(pos)
                    fit = self._fitness_model__(pos)
                    if fit < items[i][self.ID_FIT]:
                        we = items[i][self.ID_WEI]
                        items[i] = [pos, fit, rate, we]

            for i in range(self.bm_size):
                rate = 0.7 * child_pop[i][self.ID_RAT] + (child_best[self.ID_WEI] - child_pop[i][self.ID_WEI]) * \
                    np.random.uniform() * (child_best[self.ID_POS] - child_pop[i][self.ID_POS])
                pos = child_pop[i][self.ID_POS] + np.random.uniform() * rate
                pos = self._amend_solution_and_return__(pos)
                fit = self._fitness_model__(pos)
                if fit < child_pop[i][self.ID_FIT]:
                    we = child_pop[i][self.ID_WEI]
                    child_pop[i] = [pos, fit, rate, we]

            current_child_best = self._get_global_best__(child_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_child_best[self.ID_FIT] < child_best[self.ID_FIT]:
                child_best = deepcopy(current_child_best)
                if current_child_best[self.ID_FIT] < g_best[self.ID_FIT]:
                    g_best = deepcopy(current_child_best)

            for i in range(self.bm_teams):
                bt = self._get_global_best__(bm_pop[i], self.ID_FIT, self.ID_MIN_PROBLEM)
                best[i] = bt
            current_g_best = deepcopy(self._get_global_best__(best, self.ID_FIT, self.ID_MIN_PROBLEM))
            if current_g_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Fit = {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_FIT], self.loss_train
