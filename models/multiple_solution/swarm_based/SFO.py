import numpy as np
from math import gamma
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseSFO(RootAlgo):
    """
    The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm for solving
        constrained engineering optimization problems
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, sfo_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = sfo_paras["epoch"]

        self.pop_size = sfo_paras["pop_size"]       # SailFish pop size
        self.pp = sfo_paras["pp"]                   # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        self.A = sfo_paras["A"]                     # A = 4, 6,...
        self.epxilon = sfo_paras["epxilon"]         # = 0.0001, 0.001

    def _train__(self):
        s_size = int(self.pop_size / self.pp)
        sf_pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        s_pop = [self._create_solution__() for _ in range(0, s_size)]
        sf_gbest = self._get_global_best__(sf_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
        s_gbest = self._get_global_best__(s_pop, self.ID_FIT, self.ID_MIN_PROBLEM)

        for epoch in range(0, self.epoch):

            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, self.pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * np.random.uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * ( np.random.uniform() *
                                        ( sf_gbest[self.ID_POS] + s_gbest[self.ID_POS] ) / 2 - sf_pop[i][self.ID_POS] )

            ## Calculate AttackPower using Eq.(10)
            AP = self.A * ( 1 - 2 * (epoch + 1) * self.epxilon )
            if AP < 0.5:
                alpha = int(len(s_pop) * AP )
                beta = int(self.problem_size * AP)
                ### Random choice number of sardines which will be updated their position
                list1 = np.random.choice(range(0, len(s_pop)), alpha)
                for i in range(0, len(s_pop)):
                    if i in list1:
                        #### Random choice number of dimensions in sardines updated
                        list2 = np.random.choice(range(0, self.problem_size), beta)
                        for j in range(0, self.problem_size):
                            if j in list2:
                                ##### Update the position of selected sardines and selected their dimensions
                                s_pop[i][self.ID_POS][j] = np.random.uniform()*( sf_gbest[self.ID_POS][j] - s_pop[i][self.ID_POS][j] + AP )
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(0, len(s_pop)):
                    s_pop[i][self.ID_POS] = np.random.uniform()*( sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP )

            ## Recalculate the fitness of all sardine
            for i in range(0, len(s_pop)):
                s_pop[i][self.ID_FIT] = self._fitness_model__(s_pop[i][self.ID_POS], self.ID_MIN_PROBLEM)

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: temp[self.ID_FIT])
            s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
            for i in range(0, self.pop_size):
                s_size_2 = len(s_pop)
                if s_size_2 == 0:
                    s_pop = [self._create_solution__() for _ in range(0, s_size)]
                    s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
                for j in range(0, s_size):
                    ### If there is a better solution in sardine population.
                    if sf_pop[i][self.ID_FIT] > s_pop[j][self.ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size

            sf_current_best = self._get_global_best__(sf_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            s_current_best = self._get_global_best__(s_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if sf_current_best[self.ID_FIT] < sf_gbest[self.ID_FIT]:
                sf_gbest = deepcopy(sf_current_best)
            if s_current_best[self.ID_FIT] < s_gbest[self.ID_FIT]:
                s_gbest = deepcopy(s_current_best)

            self.loss_train.append(sf_gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Fit = {}".format(epoch + 1, sf_gbest[self.ID_FIT]))

        return sf_gbest[self.ID_POS], self.loss_train



class LevySFO(BaseSFO):
    """
    Levy-flight The Sailfish Optimizer - LSFO

    This version not completed yet. Need more time on this later.
    """
    def __init__(self, root_algo_paras=None, sfo_paras=None):
        BaseSFO.__init__(self, root_algo_paras, sfo_paras)

    def _levy_flight_HHO__(self, dimension=None, solution=None, prey=None):
        beta = 1.5
        sigma_muy = np.power((gamma(1+beta) * np.sin(np.pi*beta/2) / gamma((1+beta)/2)*beta*2**((beta-1)/2)) , 1.0/beta)
        u = np.random.uniform(self.domain_range[0], self.domain_range[1], dimension)
        v = np.random.uniform(self.domain_range[0], self.domain_range[1], dimension)

        LF_D =  0.01 * (sigma_muy * u) / np.power( np.abs(v), 1.0 / beta)
        LF_D2 = np.random.uniform(dimension) * LF_D
        Y = prey[self.ID_POS] + np.random.uniform(-1, 1) * np.abs( prey[self.ID_POS] - solution[self.ID_POS] )
        Z = Y + LF_D2
        return Z

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy**2)
        v = np.random.normal(0, sigma_v**2)
        s = muy / np.power(np.abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__(minmax=self.ID_MAX_PROBLEM)
        LB = 0.01 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        return levy

        #x_new = solution[0] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        #return x_new

    def _train__(self):
        s_size = int(self.pop_size / self.pp)
        sf_pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        s_pop = [self._create_solution__() for _ in range(0, s_size)]
        sf_gbest = self._get_global_best__(sf_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
        s_gbest = self._get_global_best__(s_pop, self.ID_FIT, self.ID_MIN_PROBLEM)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                if PD > 0.5:
                    #sf_pop[i][self.ID_POS] = self._levy_flight__(epoch, sf_pop[i], sf_gbest)
                    sf_pop[i][self.ID_POS] = self._levy_flight_HHO__(self.problem_size, sf_pop[i], sf_gbest)
                else:
                    lamda_i = 2 * np.random.uniform() * PD - PD
                    sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * ( np.random.uniform() *
                                        ( sf_gbest[self.ID_POS] + s_gbest[self.ID_POS] ) / 2 - sf_pop[i][self.ID_POS] )

            AP = self.A * ( 1 - 2 * (epoch + 1) * self.epxilon )
            if AP < 0.5:
                alpha = int(len(s_pop) * AP )
                beta = int(self.problem_size * AP)
                list1 = np.random.choice(range(0, len(s_pop)), alpha)
                for i in range(0, len(s_pop)):
                    if i in list1:
                        list2 = np.random.choice(range(0, self.problem_size), beta)
                        for j in range(0, self.problem_size):
                            if j in list2:
                                s_pop[i][self.ID_POS][j] = np.random.uniform()*( sf_gbest[self.ID_POS][j] - s_pop[i][self.ID_POS][j] + AP )
            else:
                for i in range(0, len(s_pop)):
                    s_pop[i][self.ID_POS] = np.random.uniform()*( sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP )
            for i in range(0, len(s_pop)):
                s_pop[i][self.ID_FIT] = self._fitness_model__(s_pop[i][self.ID_POS], self.ID_MIN_PROBLEM)

            sf_pop = sorted(sf_pop, key=lambda temp: temp[self.ID_FIT])
            s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
            for i in range(0, self.pop_size):
                s_size_2 = len(s_pop)
                if s_size_2 == 0:
                    s_pop = [self._create_solution__() for _ in range(0, s_size)]
                    s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
                for j in range(0, s_size):
                    if sf_pop[i][self.ID_FIT] > s_pop[j][self.ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break

            sf_current_best = self._get_global_best__(sf_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            s_current_best = self._get_global_best__(s_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if sf_current_best[self.ID_FIT] < sf_gbest[self.ID_FIT]:
                sf_gbest = deepcopy(sf_current_best)
            if s_current_best[self.ID_FIT] < s_gbest[self.ID_FIT]:
                s_gbest = deepcopy(s_current_best)

            self.loss_train.append(sf_gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Fit = {}".format(epoch + 1, sf_gbest[self.ID_FIT]))

        return sf_gbest[self.ID_POS], self.loss_train


class ImprovedSFO(RootAlgo):
    """
    Improved Sailfish Optimizer - ISFO
    (Actually, this version still based on Opposition-based Learning and reform Energy equation)
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, isfo_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = isfo_paras["epoch"]
        self.pop_size = isfo_paras["pop_size"]       # SailFish pop size
        self.pp = isfo_paras["pp"]         # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.1, 0.01, 0.001

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(
            gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),
            1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy**2)
        v = np.random.normal(0, sigma_v**2)
        s = muy / np.power(np.abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__(minmax=self.ID_MAX_PROBLEM)
        LB = 0.01 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        return levy

        # x_new = solution[0] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        # return x_new

    def _train__(self):
        s_size = int(self.pop_size / self.pp)
        sf_pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        s_pop = [self._create_solution__() for _ in range(0, s_size)]
        sf_gbest = self._get_global_best__(sf_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
        s_gbest = self._get_global_best__(s_pop, self.ID_FIT, self.ID_MIN_PROBLEM)

        for epoch in range(0, self.epoch):

            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, self.pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * np.random.uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * ( np.random.uniform() *
                                        ( sf_gbest[self.ID_POS] + s_gbest[self.ID_POS] ) / 2 - sf_pop[i][self.ID_POS] )

            ## Calculate AttackPower using my Eq.thieu
            #### This is our proposed, simple but effective, no need A and epxilon parameters
            AP = 1 - epoch * 1.0 / self.epoch
            for i in range(0, len(s_pop)):
                temp = ( sf_gbest[self.ID_POS] + AP ) / 2
                s_pop[i][self.ID_POS] = np.ones(self.problem_size) * self.domain_range[1] + np.ones(self.problem_size) \
                                * self.domain_range[0] - temp + np.random.uniform()*(temp - s_pop[i][self.ID_POS])

            ## Recalculate the fitness of all sardine
            for i in range(0, len(s_pop)):
                s_pop[i][self.ID_FIT] = self._fitness_model__(s_pop[i][self.ID_POS], self.ID_MIN_PROBLEM)

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: temp[self.ID_FIT])
            s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
            # t10 = 0
            for i in range(0, self.pop_size):
                # s_size_2 = len(s_pop)
                # if s_size_2 == 0:
                #     t10 += 1
                #     print(t10)
                #     s_pop = [self._create_solution__() for _ in range(0, s_size)]
                #     s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
                for j in range(0, s_size):
                    ### If there is a better solution in sardine population.
                    if sf_pop[i][self.ID_FIT] > s_pop[j][self.ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        temp = self._levy_flight__(epoch, s_pop[j], s_gbest)
                        temp = self._amend_solution_and_return__(temp)
                        fit = self._fitness_model__(temp)
                        s_pop[j] = [temp, fit]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size

            sf_current_best = self._get_global_best__(sf_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            s_current_best = self._get_global_best__(s_pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if sf_current_best[self.ID_FIT] < sf_gbest[self.ID_FIT]:
                sf_gbest = deepcopy(sf_current_best)
            if s_current_best[self.ID_FIT] < s_gbest[self.ID_FIT]:
                s_gbest = deepcopy(s_current_best)

            self.loss_train.append(sf_gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Fit = {}".format(epoch + 1, sf_gbest[self.ID_FIT]))

        return sf_gbest[self.ID_POS], self.loss_train


