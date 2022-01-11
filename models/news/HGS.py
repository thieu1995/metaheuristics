#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:45, 11/01/2022                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from models.optimizer import Optimizer


class BaseHGS(Optimizer):

    def __init__(self, obj_func=None, lb=list, ub=list, verbose=True, epoch=1000, pop_size=100, L=0.08, LH=10000):
        super().__init__(obj_func, lb, ub, verbose)
        self.epoch = epoch
        self.pop_size = pop_size
        self.L = L
        self.LH = LH

    def get_hunger_list(self, pop=None, hunger_list=np.array, g_best=None, g_worst=None):
        # min_index = pop.index(min(pop, key=lambda x: x[self.ID_FIT]))
        # Eq (2.8) and (2.9)
        for i in range(0, self.pop_size):
            r = np.random.rand()
            # space: since we pass lower bound and upper bound as list. Better take the mean of them.
            space = np.mean(self.ub - self.lb)
            H = (pop[i][self.ID_FIT] - g_best[self.ID_FIT]) / (g_worst[self.ID_FIT] - g_best[self.ID_FIT] + self.EPSILON) * r * 2 * space
            if H < self.LH:
                H = self.LH * (1 + r)
            hunger_list[i] += H

            if g_best[self.ID_FIT] == pop[i][self.ID_FIT]:
                hunger_list[i] = 0
        return hunger_list

    def sech(self, x):
        return 2 / (np.exp(x) + np.exp(-x))

    def train(self):
        # Hungry value of all solutions
        hunger_list = np.ones(self.pop_size)

        # Create population
        pop = [self.create_solution() for _ in range(self.pop_size)]

        ## Eq. (2.2)
        ### Find the current best and current worst
        g_best, g_worst = self.get_global_best_global_worst_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        hunger_list = self.get_hunger_list(pop, hunger_list, g_best, g_worst)

        # Loop
        for epoch in range(self.epoch):

            ## Eq. (2.4)
            shrink = 2 * (1 - (epoch + 1) / self.epoch)

            for i in range(0, self.pop_size):
                #### Variation control
                E = self.sech(pop[i][self.ID_FIT] - g_best[self.ID_FIT])

                # R is a ranging controller added to limit the range of activity, in which the range of R is gradually reduced to 0
                R = 2 * shrink * np.random.rand() - shrink  # Eq. (2.3)

                ## Calculate the hungry weight of each position
                if np.random.rand() < self.L:
                    W1 = hunger_list[i] * self.pop_size / (np.sum(hunger_list) + self.EPSILON) * np.random.rand()
                else:
                    W1 = 1
                W2 = (1 - np.exp(-np.abs(hunger_list[i] - np.sum(hunger_list)))) * np.random.rand() * 2

                ### Udpate position of individual Eq. (2.1)
                r1 = np.random.rand()
                r2 = np.random.rand()
                if r1 < self.L:
                    pos_new = pop[i][self.ID_POS] * (1 + np.random.normal(0, 1))
                else:
                    if r2 > E:
                        pos_new = W1 * g_best[self.ID_POS] + R * W2 * np.abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        pos_new = W1 * g_best[self.ID_POS] - R * W2 * np.abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit_new]

            ## Update global best and global worst
            g_best, g_worst = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)

            ## Update hunger list
            hunger_list = self.get_hunger_list(pop, hunger_list, g_best, g_worst)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
