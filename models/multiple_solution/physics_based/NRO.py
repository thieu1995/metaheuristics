import numpy as np
from copy import deepcopy
from math import gamma
import scipy.stats as ss
from models.multiple_solution.root_multiple import RootAlgo

class BaseNRO(RootAlgo):
    """
    An Approach Inspired from Nuclear Reaction Processes for Numerical Optimization (NRO)

    Nuclear Reaction Optimization: A novel and powerful physics-based algorithm for global optimization
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, nro_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = nro_paras["epoch"]
        self.pop_size = nro_paras["pop_size"]

    def _amend_solution_and_return__(self, solution=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0] or solution[i] > self.domain_range[1]:
                solution[i] = np.random.uniform(self.domain_range[0], self.domain_range[1])
        return solution

    def _check_array_equal__(self, array1, array2):
        check = True
        for i in range(len(array1)):
            if array1[i] != array2[i]:
                check = False
                break
        return check

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROBLEM) for _ in range(self.pop_size)]
        g_best = max(pop, key=lambda x: x[self.ID_FIT])

        for epoch in range(self.epoch):

            xichma_v = 1
            xichma_u = ((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
            levy_b = (np.random.normal(0, xichma_u ** 2)) / (np.sqrt(np.random.normal(0, xichma_v ** 2)) ** (1.0 / 1.5))

            # NFi phase
            Pb = np.random.uniform()
            Pfi = np.random.uniform()
            freq = 0.05
            alpha = 0.01
            for i in range(self.pop_size):

                ## Calculate neutron vector Nei by Eq. (2)
                ## Random 1 more index to select neutron
                temp1 = list( set(range(0, self.pop_size)) - set([i]))
                i1 = np.random.choice(temp1, replace=False)
                Nei = (pop[i][self.ID_POS] + pop[i1][self.ID_POS]) / 2
                Xi = None
                ## Update population of fission products according to Eq.(3), (6) or (9);
                if np.random.uniform() <= Pfi:
                    ### Update based on Eq. 3
                    if np.random.uniform() <= Pb:
                        xichma1 = (np.log(epoch + 1) * 1.0 / (epoch+1)) * np.abs( np.subtract(pop[i][self.ID_POS], g_best[self.ID_POS]))
                        gauss = np.array([np.random.normal(g_best[self.ID_POS][j], xichma1[j]) for j in range(self.problem_size)])
                        Xi = gauss + np.random.uniform() * g_best[self.ID_POS] - round(np.random.rand() + 1)*Nei
                    ### Update based on Eq. 6
                    else:
                        i2 = np.random.choice(temp1, replace=False)
                        xichma2 = (np.log(epoch + 1) * 1.0 / (epoch+1)) * np.abs( np.subtract(pop[i2][self.ID_POS], g_best[self.ID_POS]))
                        gauss = np.array([np.random.normal(pop[i][self.ID_POS][j], xichma2[j]) for j in range(self.problem_size)])
                        Xi = gauss + np.random.uniform() * g_best[self.ID_POS] - round(np.random.rand() + 2) * Nei
                ## Update based on Eq. 9
                else:
                    i3 = np.random.choice(temp1, replace=False)
                    xichma2 = (np.log(epoch + 1) * 1.0 / (epoch+1)) * np.abs( np.subtract(pop[i3][self.ID_POS], g_best[self.ID_POS]))
                    Xi = np.array([np.random.normal(pop[i][self.ID_POS][j], xichma2[j]) for j in range(self.problem_size)])

                ## Check the boundary and evaluate the fitness function
                Xi = self._amend_solution_and_return__(Xi)
                fit = self._fitness_model__(Xi, self.ID_MIN_PROBLEM)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [Xi, fit]
                if fit < g_best[self.ID_FIT]:
                    g_best = [Xi, fit]

            # NFu phase

            ## Ionization stage
            ## Calculate the Pa through Eq. (10);
            ranked_pop = ss.rankdata([pop[i][self.ID_FIT] for i in range(self.pop_size)])
            for i in range(self.pop_size):
                X_ion = deepcopy(pop[i][self.ID_POS])
                if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.uniform():
                    temp1 = list(set(range(0, self.pop_size)) - set([i]))
                    i1, i2 = np.random.choice(temp1, 2, replace=False)

                    for j in range(self.problem_size):
                        #### Levy flight strategy is described as Eq. 18
                        if pop[i2][self.ID_POS][j] == pop[i][self.ID_POS][j]:
                            X_ion[j] = pop[i][self.ID_POS][j] + alpha * levy_b * ( pop[i][self.ID_POS][j] - g_best[self.ID_POS][j])
                        #### If not, based on Eq. 11, 12
                        else:
                            if np.random.uniform() <= 0.5:
                                X_ion[j] = pop[i1][self.ID_POS][j] + np.random.uniform() * (pop[i2][self.ID_POS][j] - pop[i][self.ID_POS][j])
                            else:
                                X_ion[j] = pop[i1][self.ID_POS][j] - np.random.uniform() * (pop[i2][self.ID_POS][j] - pop[i][self.ID_POS][j])

                else:   #### Levy flight strategy is described as Eq. 21
                    X_worst = self._get_global_worst__(pop, self.ID_FIT, self.ID_MAX_PROBLEM)
                    for j in range(self.problem_size):
                        ##### Based on Eq. 21
                        if X_worst[self.ID_POS][j] == g_best[self.ID_POS][j]:
                            X_ion[j] = pop[i][self.ID_POS][j] + alpha * levy_b * (self.domain_range[1] - self.domain_range[0])
                        ##### Based on Eq. 13
                        else:
                            X_ion[j] = pop[i][self.ID_POS][j] + round(np.random.uniform()) * np.random.uniform()*( X_worst[self.ID_POS][j] - g_best[self.ID_POS][j] )

                ## Check the boundary and evaluate the fitness function for X_ion
                X_ion = self._amend_solution_and_return__(X_ion)
                fit = self._fitness_model__(X_ion, self.ID_MIN_PROBLEM)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_ion, fit]
                if fit < g_best[self.ID_FIT]:
                    g_best = [X_ion, fit]
            ## Fusion Stage

            ### all ions obtained from ionization are ranked based on (14) - Calculate the Pc through Eq. (14)
            ranked_pop = ss.rankdata([pop[i][self.ID_FIT] for i in range(self.pop_size)])
            for i in range(self.pop_size):

                X_fu = deepcopy(pop[i][self.ID_POS])
                temp1 = list(set(range(0, self.pop_size)) - set([i]))
                i1, i2 = np.random.choice(temp1, 2, replace=False)

                #### Generate fusion nucleus
                if (ranked_pop[i] * 1.0 / self.pop_size) < np.random.uniform():
                    t1 = np.random.uniform() * (pop[i1][self.ID_POS] - g_best[self.ID_POS])
                    t2 = np.random.uniform() * (pop[i2][self.ID_POS] - g_best[self.ID_POS])
                    temp2 = pop[i1][self.ID_POS] - pop[i2][self.ID_POS]
                    X_fu = pop[i][self.ID_POS] + t1 + t2 - np.exp(-np.linalg.norm(temp2)) * temp2
                #### Else
                else:
                    ##### Based on Eq. 22
                    if self._check_array_equal__(pop[i1][self.ID_POS], pop[i2][self.ID_POS]):
                        X_fu = pop[i][self.ID_POS] + alpha * levy_b * (pop[i][self.ID_POS] - g_best[self.ID_POS])
                    ##### Based on Eq. 16, 17
                    else:
                        if np.random.uniform() > 0.5:
                            X_fu = pop[i][self.ID_POS] - 0.5*(np.sin(2*np.pi*freq*epoch + np.pi)*(self.epoch - epoch)/self.epoch + 1)*(pop[i1][self.ID_POS] - pop[i2][self.ID_POS])
                        else:
                            X_fu = pop[i][self.ID_POS] - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) * epoch / self.epoch + 1) * (pop[i1][self.ID_POS] - pop[i2][self.ID_POS])

                X_fu = self._amend_solution_and_return__(X_fu)
                fit = self._fitness_model__(X_fu, self.ID_MIN_PROBLEM)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [X_fu, fit]
                if fit < g_best[self.ID_FIT]:
                    g_best = [X_fu, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train, g_best[self.ID_FIT]