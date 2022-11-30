#!/usr/bin/env python
# Created by "Thieu" at 19:31, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from models.optimizer import Optimizer
from copy import deepcopy


class BaseCGO(Optimizer):

    # Chaos Game Optimization: a novel metaheuristic algorithm

    def __init__(self, obj_func=None, lb=list, ub=list, verbose=True, epoch=1000, pop_size=100):
        super().__init__(obj_func, lb, ub, verbose)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        # Create population
        pop = [self.create_solution() for _ in range(self.pop_size)]

        ### Find the current best and current worst
        g_best, _ = self.get_global_best_global_worst_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Loop
        for epoch in range(self.epoch):

            for idx in range(0, self.pop_size):

                s1, s2, s3 = np.random.choice(range(0, self.pop_size), 3, replace=False)
                MG = (pop[s1][self.ID_POS] + pop[s2][self.ID_POS] + pop[s3][self.ID_POS]) / 3

                ## Calculating alpha based on Eq. 7
                alpha1 = np.random.rand()
                alpha2 = 2 * np.random.rand()
                alpha3 = 1 + np.random.random() * np.random.rand()
                esp = np.random.random()
                # There is no usage of this variable in the paper
                alpha4 = esp + esp * np.random.rand()

                beta = np.random.randint(0, 2, 3)
                gama = np.random.randint(0, 2, 3)
                ## The seed4 is mutation process, but not sure k is multiple variables or 1 variable.
                ## In the text said, multiple variables, but the defination of k is 1 variable. So confused
                k = np.random.randint(0, self.problem_size)
                k_idx = np.random.choice(range(0, self.problem_size), k, replace=False)

                seed1 = pop[idx][self.ID_POS] + alpha1 * (beta[0] * g_best[self.ID_POS] - gama[0] * MG)     # Eq. 3
                seed2 = g_best[self.ID_POS] + alpha2 * (beta[1] * pop[idx][self.ID_POS] - gama[1] * MG)     # Eq. 4
                seed3 = MG + alpha3 * (beta[2] * pop[idx][self.ID_POS] - gama[2] * g_best[self.ID_POS])     # Eq. 5
                seed4 = deepcopy(pop[idx][self.ID_POS])
                seed4[k_idx] += np.random.uniform(0, 1, k)

                # Check if solutions go outside the search space and bring them back
                seed1 = self.amend_position(seed1)
                seed2 = self.amend_position(seed2)
                seed3 = self.amend_position(seed3)
                seed4 = self.amend_position(seed4)

                sol1 = [seed1, self.get_fitness_position(seed1)]
                sol2 = [seed2, self.get_fitness_position(seed2)]
                sol3 = [seed3, self.get_fitness_position(seed3)]
                sol4 = [seed4, self.get_fitness_position(seed4)]

                ## Lots of grammar errors in this section, so confused to understand which strategy they are using
                pool = sorted([sol1, sol2, sol3, sol4, pop[idx]], key=lambda agent: agent[self.ID_FIT])
                pop[idx] = deepcopy(pool[0])

            ## Update global best and global worst
            g_best, _ = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

