"""
After spending several day to customize and optimize, I finally implement successfully the orignal verion and
    three variant version of PathFinder Algorithm.

class: BasePFA_old is the very first try to implement PathFinder, It was successfully but slow, I keep it at the end
    of this file for reader who want to know how I develop a better version.

class: BasePFA is the final version of original version of PFA
class: OPFA is an enhanced version of PFA based on Opposition-based Learning
class: LPFA is an enhanced version of PFA based on Levy-flight trajectory
class: IPFA is an improved version of PFA based on both Opposition-based Learning and Levy-flight
    (Our proposed in the paper)

Simple test with CEC14:
Lets try C1 objective function

BasePFA: after 12 loop, it reaches value 100.0
OPFA: after 10 loop, it reaches value 100.0     (a little improvement)
LPFA: after 4 loop, it reaches value 100.0      (a huge improvement)
IPFA: after 2 loop, it reaches value 100.0      (best improvement)
"""



import numpy as np
from copy import deepcopy
from math import gamma, pi
from models.multiple_solution.root_multiple import RootAlgo

class BasePFA(RootAlgo):
    """
    A new meta-heuristic optimizer: Pathfinder algorithm
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, pfa_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = pfa_paras["epoch"]
        self.pop_size = pfa_paras["pop_size"]

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        g_best = deepcopy(pop[0])
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(self.domain_range[0], self.domain_range[1]) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * np.random.uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])
                temp2 = deepcopy(pop[j][self.ID_POS])

                t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                for k in range(1, self.pop_size):
                    dist = np.linalg.norm(pop[k][self.ID_POS] - temp1)
                    t2 = alpha * np.random.uniform() * (pop[k][self.ID_POS] - temp1)
                    t3 = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                    temp2 += t2 + t3
                temp2 += t1

                ## Update members
                temp2 = self._amend_solution_and_return__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp2, fit]

            ## Update the best solution found so far (current pathfinder)
            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest_present[self.ID_FIT]:
                gbest_present = deepcopy(current_best)

            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_FIT], self.loss_train


class BasePFA_DE(RootAlgo):
    """
    A new meta-heuristic optimizer: Pathfinder algorithm
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, pfa_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = pfa_paras["epoch"]
        self.pop_size = pfa_paras["pop_size"]

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        g_best = deepcopy(pop[0])
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(self.domain_range[0], self.domain_range[1]) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * np.random.uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])

                t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                my_list_idx = np.setxor1d( np.array(range(1, self.pop_size)) , np.array([j]) )
                idx = np.random.choice(my_list_idx)
                dist = np.linalg.norm(pop[idx][self.ID_POS] - temp1)
                t2 = alpha * np.random.uniform() * (pop[idx][self.ID_POS] - temp1)
                t3 = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                temp1 += t1 + t2 + t3

                ## Update members
                temp1 = self._amend_solution_and_return__(temp1)
                fit = self._fitness_model__(temp1)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp1, fit]

            ## Update the best solution found so far (current pathfinder)
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            current_best = deepcopy(pop[self.ID_MIN_PROBLEM])
            if current_best[self.ID_FIT] < gbest_present[self.ID_FIT]:
                gbest_present = deepcopy(current_best)

            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_FIT], self.loss_train


class BasePFA_DE_Levy(RootAlgo):
    """
    A new meta-heuristic optimizer: Pathfinder algorithm
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, pfa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = pfa_paras["epoch"]
        self.pop_size = pfa_paras["pop_size"]

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy)
        v = np.random.normal(0, sigma_v)
        s = muy / np.power(np.abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__()
        LB = 0.001 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        return levy
        #x_new = solution[self.ID_POS] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        #return x_new

    def _caculate_xichma__(self, beta):
        up = gamma(1 + beta) * np.sin(pi * beta / 2)
        down = (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2))
        xich_ma_1 = np.power(up / down, 1 / beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def _shrink_encircling_Levy__(self, current_sea_lion, epoch_i, dist, c, beta=1):
        xich_ma_1, xich_ma_2 = self._caculate_xichma__(beta)
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (np.power(np.abs(b), 1 / beta)) * dist * c
        D = np.random.uniform(self.domain_range[0], self.domain_range[1], 1)
        levy = LB * D
        return (current_sea_lion - np.sqrt(epoch_i + 1) * np.sign(np.random.random(1) - 0.5)) * levy

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = deepcopy(pop[0])
        g_best = deepcopy(current_best)

        for epoch in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(self.domain_range[0], self.domain_range[1]) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = g_best[self.ID_POS] + 2 * np.random.uniform() * (g_best[self.ID_POS] - current_best[self.ID_POS]) + A
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness_model__(temp)
            current_best = deepcopy(g_best)
            if fit < g_best[self.ID_FIT]:
                g_best = [temp, fit]
            pop[0] = deepcopy([temp, fit])

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                t1 = beta * np.random.uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                idx = np.random.choice( np.setxor1d(np.array(range(1, self.pop_size)), np.array([i])) )
                dist = np.linalg.norm(pop[idx][self.ID_POS] - pop[i][self.ID_POS])
                t2 = alpha * np.random.uniform() * (pop[idx][self.ID_POS] - pop[i][self.ID_POS])
                t3 = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                temp = pop[i][self.ID_POS] + t1 + t2 + t3

                ## Update members
                temp = self._amend_solution_and_return__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            iteration_best = deepcopy(pop[self.ID_MIN_PROBLEM])

            if iteration_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(iteration_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_FIT], self.loss_train


class OPFA(BasePFA):
    def __init__(self, root_algo_paras=None, pfa_paras = None):
        BasePFA.__init__(self, root_algo_paras, pfa_paras)

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        g_best = deepcopy(pop[0])
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(self.domain_range[0], self.domain_range[1]) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * np.random.uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])
                temp2 = deepcopy(temp1)

                t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                for k in range(1, self.pop_size):
                    dist = np.linalg.norm(pop[k][self.ID_POS] - temp1)
                    t2 = alpha * np.random.uniform() * (pop[k][self.ID_POS] - temp1)
                    t3 = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                    temp2 += t2 + t3
                temp2 += t1

                ## Update members based on Opposition-based learning
                temp2 = self._amend_solution_and_return__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp2, fit]
                else:
                    C_op = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * \
                           np.ones(self.problem_size) - gbest_present[self.ID_POS] + np.random.uniform() * \
                           (gbest_present[self.ID_POS] - temp2)
                    fit_op = self._fitness_model__(C_op)
                    if fit_op < pop[j][self.ID_FIT]:
                        pop[j] = [C_op, fit_op]

            ## Update the best solution found so far (current pathfinder)
            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest_present[self.ID_FIT]:
                gbest_present = deepcopy(current_best)

            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_FIT], self.loss_train



class LPFA(BasePFA):
    def __init__(self, root_algo_paras=None, pfa_paras = None):
        BasePFA.__init__(self, root_algo_paras, pfa_paras)

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy)
        v = np.random.normal(0, sigma_v)
        s = muy / np.power(np.abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__()
        LB = 0.01 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        return levy
        #x_new = solution[self.ID_POS] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        #return x_new

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        g_best = deepcopy(pop[0])
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(self.domain_range[0], self.domain_range[1]) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * np.random.uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])
                temp2 = deepcopy(temp1)
                if np.random.uniform() < 0.5:
                    t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                    for k in range(1, self.pop_size):
                        dist = np.linalg.norm(pop[k][self.ID_POS] - temp1)
                        t2 = alpha * np.random.uniform() * (pop[k][self.ID_POS] - temp1)
                        t3 = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                        temp2 += t2 + t3
                    temp2 += t1
                else:
                    ## Using Levy-flight to boost algorithm's convergence speed
                    temp2 = self._levy_flight__(i, pop[j], gbest_present)

                ## Update members
                temp2 = self._amend_solution_and_return__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp2, fit]

            ## Update the best solution found so far (current pathfinder)
            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest_present[self.ID_FIT]:
                gbest_present = deepcopy(current_best)

            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_FIT], self.loss_train


class IPFA(LPFA):
    def __init__(self, root_algo_paras=None, pfa_paras = None):
        LPFA.__init__(self, root_algo_paras, pfa_paras)

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        g_best = deepcopy(pop[0])
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(self.domain_range[0], self.domain_range[1]) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2*np.random.uniform()*( gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])
                temp2 = deepcopy(temp1)
                if np.random.uniform() < 0.5:
                    t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                    for k in range(1, self.pop_size):
                        dist = np.linalg.norm(pop[k][self.ID_POS] - temp1)
                        t2 = alpha * np.random.uniform() * (pop[k][self.ID_POS] - temp1)
                        t3 = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                        temp2 += t2 + t3
                    temp2 += t1
                else:
                    ## Using Levy-flight to boost algorithm's convergence speed
                    temp2 = self._levy_flight__(i, pop[j], gbest_present)

                ## Update members based on Opposition-based learning
                temp2 = self._amend_solution_and_return__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp2, fit]
                else:
                    C_op = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * \
                           np.ones(self.problem_size) - gbest_present[self.ID_POS] + np.random.uniform() * \
                           (gbest_present[self.ID_POS] - temp2)
                    fit_op = self._fitness_model__(C_op)
                    if fit_op < pop[j][self.ID_FIT]:
                        pop[j] = [C_op, fit_op]

            ## Update the best solution found so far (current pathfinder)
            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest_present[self.ID_FIT]:
                gbest_present = deepcopy(current_best)

            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_FIT], self.loss_train


class BasePFA_old(RootAlgo):
    """
    A new meta-heuristic optimizer: Pathfinder algorithm (old version, dont use it)
    """
    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, pfa_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = pfa_paras["epoch"]
        self.pop_size = pfa_paras["pop_size"]

    def _train__(self):
        # Init pop and calculate fitness
        pop_past = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop_past = sorted(pop_past, key=lambda temp: temp[self.ID_FIT])
        gbest_past = deepcopy(pop_past[0])
        gbest_present = deepcopy(gbest_past)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(self.domain_range[0], self.domain_range[1]) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2*np.random.uniform()*( gbest_present[self.ID_POS] - gbest_past[self.ID_POS]) + A
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness_model__(temp)
            gbest_past = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop_past[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            pop_new = deepcopy(pop_past)
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop_new[j][self.ID_POS])
                temp2 = deepcopy(temp1)
                t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                for k in range(1, self.pop_size):
                    dist = np.linalg.norm(pop_past[k][self.ID_POS] - temp1)
                    t2 = alpha*np.random.uniform()* (pop_past[k][self.ID_POS] - temp1)
                    t3 = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i+1)*1.0/self.epoch)* dist
                    temp2 += t2 + t3
                temp2 += t1
                pop_new[j][self.ID_POS] = self._amend_solution_and_return__(temp2)
                pop_new[j][self.ID_FIT] = self._fitness_model__(temp2)

            ## Find the best fitness (current pathfinder)
            current_best = self._get_global_best__(pop_new, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest_present[self.ID_FIT]:
                gbest_present = deepcopy(current_best)
                pop_past[0] = deepcopy(current_best)

            ## Update members if fitness better
            for j in range(1, self.pop_size):
                if pop_new[j][self.ID_FIT] < pop_past[j][self.ID_FIT]:
                    pop_past[j] = deepcopy(pop_new[j])
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_FIT], self.loss_train