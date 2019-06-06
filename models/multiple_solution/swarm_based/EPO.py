import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseEPO(RootAlgo):
    """
    Paper: Emperor penguin optimizer: A bio-inspired algorithm for engineering problems
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, epo_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch =  epo_paras["epoch"]
        self.pop_size = epo_paras["pop_size"]

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)

        M = 2
        for i in range(self.epoch):
            R = np.random.random_sample()
            T = 0 if R > 0.5 else 1
            T_s = T - self.epoch / (i - self.epoch)

            for j in range(self.pop_size):
                for k in range(self.problem_size):

                    P_grid = np.abs( gbest[self.ID_POS][k] - pop[j][self.ID_POS][k] )
                    A = M * (T_s + P_grid) * np.random.random_sample() - T_s
                    C = np.random.random_sample()

                    f = np.random.uniform(2, 3)
                    l = np.random.uniform(1.5, 2)
                    S_A = f * np.exp(-i / l) - np.exp(-i)

                    D_ep = np.abs( S_A * gbest[self.ID_POS][k] - C * pop[j][self.ID_POS][k] )

                    pop[j][self.ID_POS][k] -= A * D_ep

                self._amend_solution__(pop[j][self.ID_POS])

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest[self.ID_FIT]))

        return gbest[self.ID_FIT], self.loss_train

