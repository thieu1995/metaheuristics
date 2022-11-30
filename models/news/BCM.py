#!/usr/bin/env python
# Created by "Thieu" at 08:40, 21/02/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from models.optimizer import Optimizer


class BaseBCM(Optimizer):
    """
        BARYCENTER METHOD (BCM)
    Link: https://arxiv.org/pdf/1801.10533.pdf
    """

    def __init__(self, obj_func=None, lb=list, ub=list, verbose=True, epoch=1000, pop_size=100, nu=2, sigma=0.5, zeta=0, lamda=1):
        super().__init__(obj_func, lb, ub, verbose)
        self.epoch = epoch  # M in the paper
        self.pop_size = pop_size
        self.nu = nu            # Positive value
        self.sigma = sigma      # Std deviation of normal distribution
        self.zeta = zeta        # Proportional value for mean of normal distribution
        self.lamda = lamda      # Forgetting factor between 0 and 1

    def train(self):
        delta_x = np.zeros(self.problem_size)
        g_best = self.create_solution()
        x_initial, fit_initial = g_best[0], g_best[1]
        m_1 = 0.01
        # Loop
        for epoch in range(self.epoch):
            z = np.random.normal(self.zeta * delta_x, self.sigma)
            x_new = x_initial + z
            fit_new = self.get_fitness_position(x_new)
            e_i = np.exp(-self.nu * fit_new)
            m = self.lamda * m_1 + e_i
            x_new = (1.0/m) * (self.lamda * m_1 * x_initial + x_new * e_i)
            m_1 = m
            x_initial = x_new

            # Check if solutions go outside the search space and bring them back
            pos_new = self.amend_position(x_new)
            fit_new = self.get_fitness_position(pos_new)
            g_best = [pos_new, fit_new]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

