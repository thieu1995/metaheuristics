#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:44, 11/01/2022                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from copy import deepcopy
import numpy as np


class Optimizer:
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0  # min problem
    ID_MAX_PROB = -1  # max problem

    ID_POS = 0  # Position
    ID_FIT = 1  # Fitness

    EPSILON = 10E-10  # Avoid division by 0

    DEFAULT_LB = -1
    DEFAULT_UB = 1

    def __init__(self, obj_func=None, lb=list, ub=list, verbose=True):
        self.verbose = verbose
        self.obj_func = obj_func
        if lb is None:
            print("Lower bound need to be a list.")
            exit(0)
        else:
            self.lb = np.array(lb)
        if ub is None:
            print("Upper bound need to be a list.")
            exit(0)
        else:
            self.ub = np.array(ub)
        if len(lb) != len(ub):
            print("Lower bound and Upper bound need to have the same length")
            exit(0)
        self.problem_size = len(lb)
        self.solution, self.loss_train = None, []

    def create_solution(self, minmax=0):
        """ Return solution with 2 element: position of solution and fitness of solution
        Parameters
        ----------
        minmax
            0 - minimum problem, else - maximum problem
        """
        position = np.random.uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        return [position, fitness]

    def get_fitness_position(self, position=None, minmax=0):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :param minmax: 0- min problem, 1 - max problem
        :return:
        """
        return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + self.EPSILON)  # Avoid division by 0

    def get_fitness_solution(self, solution=None, minmax=0):
        return self.get_fitness_position(solution[self.ID_POS], minmax)

    def get_global_best_global_worst_solution(self, pop=None, id_fit=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        if id_best == self.ID_MIN_PROB:
            return deepcopy(sorted_pop[id_best]), deepcopy(sorted_pop[self.ID_MAX_PROB])
        elif id_best == self.ID_MAX_PROB:
            return deepcopy(sorted_pop[id_best]), deepcopy(sorted_pop[self.ID_MIN_PROB])

    def update_global_best_global_worst_solution(self, pop=None, id_best=None, id_worst=None, g_best=None):
        """ Sort the copy of population and update the current best position. Return the new current best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)
        return g_best, sorted_pop[id_worst]

    def amend_position(self, position=None):
        return np.clip(position, self.lb, self.ub)

    def amend_position_random(self, position=None):
        return np.where(np.logical_and(self.lb <= position, position <= self.ub), position, np.random.uniform(self.lb, self.ub))


