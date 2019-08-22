import numpy as np
from copy import deepcopy
from math import gamma
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
        pop_past = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop_past = sorted(pop_past, key=lambda temp: temp[self.ID_FIT])
        gbest_past = deepcopy(pop_past[0])
        gbest_present = deepcopy(gbest_past)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(-1, 1, self.problem_size) * np.exp(-2 * (i+1) / self.epoch)

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
                    t3 = np.random.uniform(-1, 1, self.problem_size) * (1 - (i+1)*1.0/self.epoch)* dist
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


class OPFA(BasePFA):
    def __init__(self, root_algo_paras=None, pfa_paras = None):
        BasePFA.__init__(self, root_algo_paras, pfa_paras)

    def _train__(self):
        # Init pop and calculate fitness
        pop_past = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop_past = sorted(pop_past, key=lambda temp: temp[self.ID_FIT])
        gbest_past = deepcopy(pop_past[0])
        gbest_present = deepcopy(gbest_past)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(-1, 1) * np.exp(-2 * (i+1) / self.epoch)

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
                    dist = np.linalg.norm(pop_new[k][self.ID_POS] - temp1)
                    t2 = alpha * np.random.uniform() * (pop_new[k][self.ID_POS] - temp1)
                    t3 = np.random.uniform(-1, 1, self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
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
                else:
                    C_op = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * \
                           np.ones(self.problem_size) - gbest_present[self.ID_POS] + np.random.uniform() * \
                           (gbest_present[self.ID_POS] - pop_past[j][self.ID_POS])
                    fit_op = self._fitness_model__(C_op)
                    if fit_op < pop_past[j][self.ID_FIT]:
                        pop_past[j] = [C_op, fit_op]

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
        pop_past = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop_past = sorted(pop_past, key=lambda temp: temp[self.ID_FIT])
        gbest_past = deepcopy(pop_past[0])
        gbest_present = deepcopy(gbest_past)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(-1, 1) * np.exp(-2 * (i+1) / self.epoch)

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
                if np.random.uniform() < 0.5:
                    t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                    for k in range(1, self.pop_size):
                        dist = np.linalg.norm(pop_new[k][self.ID_POS] - temp1)
                        t2 = alpha * np.random.uniform() * (pop_new[k][self.ID_POS] - temp1)
                        t3 = np.random.uniform(-1, 1, self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                        temp2 += t2 + t3
                    temp2 += t1
                else:
                    temp2 = self._levy_flight__(i, pop_new[j], gbest_present)
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



class IPFA(LPFA):
    def __init__(self, root_algo_paras=None, pfa_paras = None):
        LPFA.__init__(self, root_algo_paras, pfa_paras)

    def _train__(self):
        # Init pop and calculate fitness
        pop_past = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop_past = sorted(pop_past, key=lambda temp: temp[self.ID_FIT])
        gbest_past = deepcopy(pop_past[0])
        gbest_present = deepcopy(gbest_past)

        for i in range(self.epoch):
            alpha, beta = np.random.uniform(1, 2, 2)
            A = np.random.uniform(-1, 1) * np.exp(-2 * (i+1) / self.epoch)

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
                if np.random.uniform() < 0.5:
                    t1 = beta * np.random.uniform() * (gbest_present[self.ID_POS] - temp1)
                    for k in range(1, self.pop_size):
                        dist = np.linalg.norm(pop_new[k][self.ID_POS] - temp1)
                        t2 = alpha * np.random.uniform() * (pop_new[k][self.ID_POS] - temp1)
                        t3 = np.random.uniform(-1, 1, self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                        temp2 += t2 + t3
                    temp2 += t1
                else:
                    temp2 = self._levy_flight__(i, pop_new[j], gbest_present)
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
                else:
                    C_op = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * \
                           np.ones(self.problem_size) - gbest_present[self.ID_POS] + np.random.uniform() * \
                           (gbest_present[self.ID_POS] - pop_past[j][self.ID_POS])
                    fit_op = self._fitness_model__(C_op)
                    if fit_op < pop_past[j][self.ID_FIT]:
                        pop_past[j] = [C_op, fit_op]

            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_FIT], self.loss_train
