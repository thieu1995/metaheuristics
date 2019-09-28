import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseSOA(RootAlgo):
    """
    My version: I changed some equation and make this algorithm works
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, soa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = soa_paras["epoch"]
        self.pop_size = soa_paras["pop_size"]

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
        C_f = 1.0

        # Epoch loop
        for epoch in range(self.epoch):

            ## Each individual loop
            for i in range(self.pop_size):

                ### Based on Eq.5, 6, 7, 8, 9
                C_sp = (C_f - epoch * (C_f/self.epoch)) * pop[i][self.ID_POS]
                M_sp = np.random.uniform() * ( g_best[self.ID_POS] - pop[i][self.ID_POS] )
                D_sp = C_sp + M_sp

                ### Based on Eq. 10, 11, 12, 13, 14
                r = np.exp(np.random.uniform(0, 2*np.pi))
                temp = r * (np.sin(np.random.uniform(0, 2*np.pi)) + np.cos(np.random.uniform(0, 2*np.pi)) + np.random.uniform(0, 2*np.pi))
                P_sp = (D_sp * temp) * g_best[self.ID_POS]

                P_sp = self._amend_solution_and_return__(P_sp)
                fit = self._fitness_model__(P_sp, self.ID_MIN_PROBLEM)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [P_sp, fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [P_sp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch+1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train



class OriginalSOA(RootAlgo):
    """
    Sandpiper optimization algorithm: a novel approach for solving real-life engineering problems
    Same as : A bio-inspired based optimization algorithm for industrial engineering problems.
    Cant even update its position - completely trash
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, soa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = soa_paras["epoch"]
        self.pop_size = soa_paras["pop_size"]

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
        C_f = 2

        # Epoch loop
        for epoch in range(self.epoch):

            ## Each individual loop
            for i in range(self.pop_size):

                ### Based on Eq.5, 6, 7, 8, 9
                C_sp = (C_f - epoch * (C_f / self.epoch)) * pop[i][self.ID_POS]
                M_sp = 0.5 * np.random.uniform() * ( g_best[self.ID_POS] - pop[i][self.ID_POS] )
                D_sp = C_sp + M_sp

                ### Based on Eq. 10, 11, 12, 13, 14
                r = np.exp(np.random.uniform(0, 2*np.pi))
                temp = r * (np.sin(np.random.uniform(0, 2*np.pi)) + np.cos(np.random.uniform(0, 2*np.pi)) + np.random.uniform(0, 2*np.pi))
                P_sp = (D_sp * temp) * g_best[self.ID_POS]
                fit = self._fitness_model__(P_sp, self.ID_MIN_PROBLEM)
                pop[i] = [P_sp, fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [P_sp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch+1, g_best[self.ID_FIT]))

        return self.loss_train, g_best[self.ID_POS]

