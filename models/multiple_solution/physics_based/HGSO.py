import numpy as np
from copy import deepcopy
from math import gamma
from numpy.random import uniform
from models.multiple_solution.root_multiple import RootAlgo

class BaseHGSO(RootAlgo):
    """
    Henry gas solubility optimization: A novel physics-based algorithm
    """
    ID_POS = 0
    ID_FIT = 1
    ID_CLUS = 2     # cluster number

    def __init__(self, root_algo_paras=None, hgso_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = hgso_paras["epoch"]
        self.pop_size = hgso_paras["pop_size"]
        self.n_clusters = hgso_paras["n_clusters"]
        self.n_elements = int(self.pop_size / self.n_clusters)

    def _create_population__(self, minmax=0, n_clusters=0):
        pop = []
        group = []
        for i in range(n_clusters):
            team = []
            for j in range(self.n_elements):
                solution = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fitness = self._fitness_model__(solution=solution, minmax=minmax)
                team.append([solution, fitness, i])
                pop.append([solution, fitness, i])
            group.append(team)
        return pop, group

    def _get_best_solution_in_team(self, group=None):
        list_best = []
        for i in range(len(group)):
            sorted_team = sorted(group[i], key=lambda temp: temp[self.ID_FIT])
            list_best.append( deepcopy(sorted_team[self.ID_MIN_PROBLEM]) )
        return list_best

    def _train__(self):
        T0 = 298.15
        K = 1.0
        beta = 1.0
        alpha = 1
        epxilon = 0.05

        l1 = 5E-2
        l2 = 100.0
        l3 = 1E-2
        H_j = l1 * uniform()
        P_ij = l2 * uniform()
        C_j = l3 * uniform()

        pop, group = self._create_population__(self.ID_MIN_PROBLEM, self.n_clusters)
        g_best = max(pop, key=lambda x: x[self.ID_FIT])     # single element
        p_best = self._get_best_solution_in_team(group)     # multiple element

        # Loop iterations
        for epoch in range(self.epoch):

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    F = -1.0 if uniform() < 0.5 else 1.0

                    ##### Based on Eq. 8, 9, 10
                    H_j = H_j * np.exp(-C_j * ( 1.0/np.exp(-epoch/self.epoch) - 1.0/T0 ))
                    S_ij = K * H_j * P_ij
                    gama = beta * np.exp(- ((p_best[i][self.ID_FIT] + epxilon) / (group[i][j][self.ID_FIT] + epxilon)))

                    X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                        F * uniform() * alpha * (S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])

                    fit = self._fitness_model__(X_ij, self.ID_MIN_PROBLEM)
                    group[i][j] = [X_ij, fit, i]
                    pop[i*self.n_elements + j] = [X_ij, fit, i]

            ## Update Henry's coefficient using Eq.8
            H_j = H_j * np.exp(-C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / T0))
            ## Update the solubility of each gas using Eq.9
            S_ij = K * H_j * P_ij
            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = np.argsort([ x[self.ID_FIT] for x in pop ])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROBLEM)
                pop[id] = [X_new, fit, i]
                group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            current_best = min(pop, key=lambda x: x[self.ID_FIT])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train


class OppoHGSO(BaseHGSO):

    def _train__(self):
        T0 = 298.15
        K = 1.0
        beta = 1.0
        alpha = 1
        epxilon = 0.05

        l1 = 5E-2
        l2 = 100.0
        l3 = 1E-2
        H_j = l1 * uniform()
        P_ij = l2 * uniform()
        C_j = l3 * uniform()

        pop, group = self._create_population__(self.ID_MIN_PROBLEM, self.n_clusters)
        g_best = max(pop, key=lambda x: x[self.ID_FIT])     # single element
        p_best = self._get_best_solution_in_team(group)     # multiple element

        # Loop iterations
        for epoch in range(self.epoch):

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    ##### Based on Eq. 8, 9, 10
                    H_j = H_j * np.exp(-C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / T0))
                    S_ij = K * H_j * P_ij
                    F = -1.0 if uniform() < 0.5 else 1.0
                    gama = beta * np.exp(- ((p_best[i][self.ID_FIT] + epxilon) / (group[i][j][self.ID_FIT] + epxilon)))

                    X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                        F * uniform() * alpha * (S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])
                    X_ij = self._amend_solution_and_return__(X_ij)

                    fit = self._fitness_model__(X_ij, self.ID_MIN_PROBLEM)
                    group[i][j] = [X_ij, fit, i]
                    pop[i*self.n_elements + j] = [X_ij, fit, i]

            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = np.argsort([ x[self.ID_FIT] for x in pop ])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROBLEM)
                if fit < pop[id][self.ID_FIT]:
                    pop[id] = [X_new, fit, i]
                    group[i][j] = [X_new, fit, i]
                else:
                    t1 = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * np.ones(self.problem_size)
                    t2 = -1 * g_best[self.ID_POS] + uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    C_op = t1 + t2
                    C_op = self._amend_solution_and_return__(C_op)
                    fit_op = self._fitness_model__(C_op, self.ID_MIN_PROBLEM)
                    if fit_op < pop[id][self.ID_FIT]:
                        pop[id] = [X_new, fit, i]
                        group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            current_best = min(pop, key=lambda x: x[self.ID_FIT])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train


class LevyHGSO(BaseHGSO):

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy)
        v = np.random.normal(0, sigma_v)
        s = muy / np.power(np.abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__(minmax=self.ID_MAX_PROBLEM)
        LB = 0.001 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        #return levy

        x_new = solution[0] + 1.0/np.sqrt(epoch+1) * np.sign(np.random.uniform() - 0.5) * levy
        return x_new

    def _levy_flight_2__(self, solution=None, g_best=None):
        alpha = 0.01
        xichma_v = 1
        xichma_u = ((gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (np.random.normal(0, xichma_u ** 2)) / (np.sqrt(np.random.normal(0, xichma_v ** 2)) ** (1.0 / 1.5))
        return solution[self.ID_POS] + alpha * levy_b * (solution[self.ID_POS] - g_best[self.ID_POS])

    def _train__(self):
        T0 = 298.15
        K = 1.0
        beta = 1.0
        alpha = 1
        epxilon = 0.05

        l1 = 5E-2
        l2 = 100.0
        l3 = 1E-2
        H_j = l1 * uniform()
        P_ij = l2 * uniform()
        C_j = l3 * uniform()

        pop, group = self._create_population__(self.ID_MIN_PROBLEM, self.n_clusters)
        g_best = max(pop, key=lambda x: x[self.ID_FIT])     # single element
        p_best = self._get_best_solution_in_team(group)     # multiple element

        # Loop iterations
        for epoch in range(self.epoch):

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    ##### Based on Levy
                    if uniform() < 0.5:
                        X_ij = self._levy_flight__(epoch+1, group[i][j], g_best)
                        #X_ij = self._levy_flight_2__(group[i][j], g_best)
                    else:   ##### Based on Eq. 8, 9, 10
                        H_j = H_j * np.exp(-C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / T0))
                        S_ij = K * H_j * P_ij
                        F = -1.0 if uniform() < 0.5 else 1.0
                        gama = beta * np.exp(- ((p_best[i][self.ID_FIT] + epxilon) / (group[i][j][self.ID_FIT] + epxilon)))

                        X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                            F * uniform() * alpha * (S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])

                    X_ij = self._amend_solution_and_return__(X_ij)
                    fit = self._fitness_model__(X_ij, self.ID_MIN_PROBLEM)
                    group[i][j] = [X_ij, fit, i]
                    pop[i*self.n_elements + j] = [X_ij, fit, i]

            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = np.argsort([ x[self.ID_FIT] for x in pop ])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROBLEM)
                if fit < pop[id][self.ID_FIT]:
                    pop[id] = [X_new, fit, i]
                    group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            current_best = min(pop, key=lambda x: x[self.ID_FIT])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train


# class ITWO(OppoTWO, LevyTWO):
#     def __init__(self, root_algo_paras=None, two_paras = None):
#         OppoTWO.__init__(self, root_algo_paras, two_paras)
#
#     def _train__(self):
#
#         pop = [self._create_solution__(minmax=self.ID_MAX_PROBLEM) for _ in range(self.pop_size)]
#         g_best = max(pop, key=lambda x: x[self.ID_FIT])
#         pop_new = deepcopy(pop)
#         for epoch in range(self.epoch):
#             for i in range(self.pop_size):
#                 if np.random.uniform() < 0.5:
#                     pop_new[i][self.ID_POS] = self._levy_flight__(epoch + 1, pop_new[i], g_best)
#                 else:
#                     for j in range(self.pop_size):
#                         if pop[i][self.ID_WEIGHT] < pop[j][self.ID_WEIGHT]:
#                             force = max(pop[i][self.ID_WEIGHT] * muy_s, pop[j][self.ID_WEIGHT] * muy_s)
#                             resultant_force = force - pop[i][self.ID_WEIGHT] * muy_k
#                             g = pop[j][self.ID_POS] - pop[i][self.ID_POS]
#                             acceleration = resultant_force * g / (pop[i][self.ID_WEIGHT] * muy_k)
#                             delta_x = 1 / 2 * acceleration + np.power(alpha, epoch + 1) * beta * \
#                                       (self.domain_range[1] - self.domain_range[0]) * np.random.randn(self.problem_size)
#                             pop_new[i][self.ID_POS] += delta_x
#
#             pop_new = self._amend_and_return_pop__(pop, pop_new, g_best, epoch + 1)
#             pop_new = self._update_fit__(pop_new)
#
#             for i in range(self.pop_size):
#                 if pop[i][self.ID_FIT] < pop_new[i][self.ID_FIT]:
#                     pop[i] = deepcopy(pop_new[i])
#                 else:
#                     C_op = self.domain_range[1] * np.ones(self.problem_size) + self.domain_range[0] * np.ones(self.problem_size) + \
#                            -1 * g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
#                     fit_op = self._fitness_model__(C_op, self.ID_MAX_PROBLEM)
#                     if fit_op > pop[i][self.ID_FIT]:
#                         pop[i] = [C_op, fit_op, 0.0]
#             pop = self._update_weight__(pop)
#             pop_new = deepcopy(pop)
#
#             current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MAX_PROBLEM)
#             if current_best[self.ID_FIT] > g_best[self.ID_FIT]:
#                 g_best = deepcopy(current_best)
#             self.loss_train.append(1.0 / g_best[self.ID_FIT])
#             if self.print_train:
#                 print("Generation : {0}, best result so far: {1}".format(epoch + 1, 1.0 / g_best[self.ID_FIT]))
#         return g_best[self.ID_POS], self.loss_train
#
