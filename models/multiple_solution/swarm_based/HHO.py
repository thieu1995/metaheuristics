from copy import deepcopy
import numpy as np
from math import gamma
from models.multiple_solution.root_multiple import RootAlgo

class BaseHHO(RootAlgo):
    """
    Harris Hawks Optimization: Algorithm and Applications
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, hho_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = hho_paras["epoch"]
        self.pop_size = hho_paras["pop_size"]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        gbest = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
        for epoch in range(0, self.epoch):

            # Update the location of Harris' hawks
            for i in range(0, self.pop_size):
                E0 = 2 * np.random.uniform() - 1                        # -1 < E0 < 1
                E = 2 * E0 * (1 - (epoch + 1) * 1.0 / self.epoch)       # factor to show the decreasing energy of rabbit
                J = 2 * (1 - np.random.uniform())

                # -------- Exploration phase Eq. (1) in paper -------------------
                if (np.abs(E) >= 1):
                    # Harris' hawks perch randomly based on 2 strategy:
                    if (np.random.uniform() >= 0.5):        # perch based on other family members
                        X_rand = deepcopy(pop[np.random.randint(0, self.pop_size)][self.ID_POS])
                        pop[i][self.ID_POS] = X_rand - np.random.uniform() * np.abs(X_rand - 2 * np.random.uniform() * pop[i][self.ID_POS])

                    else:           # perch on a random tall tree (random site inside group's home range)
                        X_m = np.mean([x[self.ID_POS] for x in pop])
                        pop[i][self.ID_POS] = (gbest[self.ID_POS] - X_m) - np.random.uniform()*(
                            self.domain_range[0] + np.random.uniform() * (self.domain_range[1] - self.domain_range[0]))

                # -------- Exploitation phase -------------------
                else:
                    # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                    # phase 1: ----- surprise pounce (seven kills) ----------
                    # surprise pounce (seven kills): multiple, short rapid dives by different hawks
                    if (np.random.uniform() >= 0.5):
                        delta_X = gbest[self.ID_POS] - pop[i][self.ID_POS]
                        if (np.abs(E) >= 0.5):          # Hard besiege Eq. (6) in paper
                            pop[i][self.ID_POS] = delta_X - E * np.abs( J * gbest[self.ID_POS] - pop[i][self.ID_POS] )
                        else:                           # Soft besiege Eq. (4) in paper
                            pop[i][self.ID_POS] = gbest[self.ID_POS] - E * np.abs(delta_X)
                    else:
                        xichma = np.power((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2.0)) / (gamma((1 + 1.5) * 1.5 * np.power(2, (1.5 - 1) / 2)) / 2.0), 1.0 / 1.5)
                        LF_D = 0.01 * np.random.uniform() * xichma / np.power(np.abs(np.random.uniform()), 1.0 / 1.5)
                        fit_Y, Y = None, None
                        if (np.abs(E) >= 0.5):      # Soft besiege Eq. (10) in paper
                            Y = gbest[self.ID_POS] - E * np.abs( J * gbest[self.ID_POS] - pop[i][self.ID_POS] )
                            fit_Y = self._fitness_model__(Y)
                        else:                       # Hard besiege Eq. (11) in paper
                            X_m = np.mean([x[self.ID_POS] for x in pop])
                            Y = gbest[self.ID_POS] - E * np.abs( J * gbest[self.ID_POS] - X_m )
                            fit_Y = self._fitness_model__(Y)

                        Z = Y + np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * LF_D
                        fit_Z = self._fitness_model__(Z)

                        if fit_Y < pop[i][self.ID_FIT]:
                            pop[i] = [Y, fit_Y]
                        if fit_Z < pop[i][self.ID_FIT]:
                            pop[i] = [Z, fit_Z]

            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Fit = {}".format(epoch + 1, gbest[self.ID_FIT]))

        return gbest[self.ID_POS], self.loss_train

