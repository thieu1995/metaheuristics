#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:15, 12/01/2022                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from models.optimizer import Optimizer


class BaseRUN(Optimizer):

    def __init__(self, obj_func=None, lb=list, ub=list, verbose=True, epoch=1000, pop_size=100, pr=0.5):
        super().__init__(obj_func, lb, ub, verbose)
        self.epoch = epoch  # M in the paper
        self.pop_size = pop_size
        self.pr = pr  # Probability Parameter
        self.beta = [0.2, 1.2]

    def train(self):
        # Create population
        pop = [self.create_solution() for _ in range(self.pop_size)]

        ### Find the current best and current worst
        g_best, g_worst = self.get_global_best_global_worst_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Loop
        for epoch in range(self.epoch):

            # Eq.(14.2), Eq.(14.1)
            beta = self.beta[0] + (self.beta[1] - self.beta[0]) * (1 - ((epoch + 1) / self.epoch) ** 3) ** 2
            alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))

            for idx in range(0, self.pop_size):

                p1 = 2 * np.random.rand() * alpha - alpha
                p2 = 2 * np.random.rand() * alpha - alpha

                #  Four positions randomly selected from population
                r1, r2, r3, r4 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
                # Average of Four positions randomly selected from population
                r0 = (pop[r1][self.ID_POS] + pop[r2][self.ID_POS] + pop[r3][self.ID_POS] + pop[r4][self.ID_POS]) / 4
                # Randomization Epsilon
                epsilon = 5e-3 * np.random.rand()

                delta = 2 * np.random.rand() * np.abs(r0 - pop[idx][self.ID_POS])
                step = (g_best[self.ID_POS] - pop[r1][self.ID_POS] + delta) / 2
                delta_x = np.random.choice(range(0, self.pop_size)) * np.abs(step)
                x1 = pop[idx][self.ID_POS] - np.random.normal() * p1 * 2 * delta_x * pop[idx][self.ID_POS] / (
                            g_worst[self.ID_POS] - g_best[self.ID_POS] + epsilon) + \
                     np.random.rand() * p2 * (g_best[self.ID_POS] - pop[idx][self.ID_POS])

                z = pop[idx][self.ID_POS] - np.random.normal() * 2 * delta_x * pop[idx][self.ID_POS] / (g_worst[self.ID_POS] - g_best[self.ID_POS] + epsilon)
                y_p = np.random.rand() * ((z + pop[idx][self.ID_POS]) / 2 + np.random.rand() * delta_x)
                y_q = np.random.rand() * ((z + pop[idx][self.ID_POS]) / 2 - np.random.rand() * delta_x)
                x2 = g_best[self.ID_POS] - np.random.normal() * p1 * 2 * delta_x * pop[idx][self.ID_POS] / (y_p - y_q + epsilon) + \
                     np.random.rand() * p2 * (pop[r1][self.ID_POS] - pop[r2][self.ID_POS])

                x3 = pop[idx][self.ID_POS] - p1 * (x2 - x1)
                ra = np.random.uniform(0, 1, self.problem_size)
                rb = np.random.rand(0, 1, self.problem_size)
                pos_new = ra * (rb * x1 + (1 - rb) * x2) + (1 - ra) * x3

                # Local escaping operator
                if np.random.rand() < self.pr:
                    f1 = np.random.uniform(-1, 1)
                    f2 = np.random.normal(0, 1)
                    L1 = np.round(1 - np.random.rand())
                    u1 = L1 * 2 * np.random.rand() + (1 - L1)
                    u2 = L1 * np.random.rand() + (1 - L1)
                    u3 = L1 * np.random.rand() + (1 - L1)

                    L2 = np.round(1 - np.random.rand())
                    x_rand = np.random.uniform(self.lb, self.ub)
                    x_p = pop[np.random.choice(range(0, self.pop_size))][self.ID_POS]
                    x_m = L2 * x_p + (1 - L2) * x_rand

                    if np.random.rand() < 0.5:
                        pos_new = pos_new + f1 * (u1 * g_best[self.ID_POS] - u2 * x_m) + f2 * p1 * (
                                    u3 * (x2 - x1) + u2 * (pop[r1][self.ID_POS] - pop[r2][self.ID_POS])) / 2
                    else:
                        pos_new = g_best[self.ID_POS] + f1 * (u1 * g_best[self.ID_POS] - u2 * x_m) + f2 * p1 * (
                                    u3 * (x2 - x1) + u2 * (pop[r1][self.ID_POS] - pop[r2][self.ID_POS])) / 2

                # Check if solutions go outside the search space and bring them back
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                pop[idx] = [pos_new, fit_new]

            ## Update global best and global worst
            g_best, g_worst = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
