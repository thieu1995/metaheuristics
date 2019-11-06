import numpy as np
from copy import deepcopy
import math
from copy import copy
from models.multiple_solution.root_multiple import RootAlgo


class SLnO(RootAlgo):

    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, woa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = woa_paras["epoch"]
        self.pop_size = woa_paras["pop_size"]

    def _evaluate_population__(self, solution):

        # population = np.maximum(population, range0)
        # population = np.minimum(population, range1)
        for j in range(self.problem_size):
            if solution[j] > self.domain_range[1] \
                    or float(solution[j]) == float(self.domain_range[1]):
                solution[j] = np.random.uniform(self.domain_range[1] - 2, self.domain_range[1])

            if solution[j] < self.domain_range[0] or \
                    float(solution[j]) == float(self.domain_range[0]):
                solution[j] = np.random.uniform(self.domain_range[0], self.domain_range[0] + 2)

        return solution

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        # Find prey which is the best solution
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)

        for i in range(self.epoch):

            for j in range(self.pop_size):

                SP_leader = np.random.uniform(0, 1)
                if SP_leader >= 0.6:
                    m = np.random.uniform(-1, 1)
                    new_position = np.abs(gbest[self.ID_POS] - pop[j][self.ID_POS]) * np.cos(2 * np.pi * m) \
                                        + gbest[self.ID_POS]
                else:
                    c = 2 - 2 * i / self.epoch
                    b = np.random.uniform(0, 1, self.problem_size)
                    p = np.random.uniform(0, 1)

                    if c <= 1:
                        dist = b * np.abs(2 * gbest[self.ID_POS] - pop[j][self.ID_POS])
                        new_position = gbest[self.ID_POS] - dist*c
                    else:

                        rand_index = np.random.randint(0, self.pop_size)
                        random_SL = pop[rand_index]

                        dist = np.abs(b * random_SL[self.ID_POS] - pop[j][self.ID_POS])
                        new_position = random_SL[self.ID_POS] - dist*c
                # new_position[new_position < self.domain_range[0]] = self.domain_range[0]
                # new_position[new_position > self.domain_range[1]] = self.domain_range[1]

                new_position = self._evaluate_population__(new_position)
                fit = self._fitness_model__(new_position)
                pop[j] = [new_position, fit]

            current_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Best fit so far = {}".format(i + 1, gbest[self.ID_FIT]))

        return gbest[self.ID_POS], self.loss_train


class ISLO(SLnO):

    ID_POS = 0
    ID_FIT = 1
    ID_PBEST_POS = 2
    ID_PBEST_FIT = 3

    def __init__(self, root_algo_paras=None, woa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = woa_paras["epoch"]
        self.pop_size = woa_paras["pop_size"]

    def _create_solution__(self, minmax=0):

        x = np.random.uniform(self.domain_range[0], self.domain_range[1], (self.problem_size, 1))
        x_past_best = deepcopy(x)
        x_fitness = self._fitness_model__(x, minmax=minmax)
        x_past_fitness = deepcopy(x_fitness)
        return [x, x_fitness,  x_past_best, x_past_fitness]

    def _caculate_xichma__(self, beta):
        up = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        down = (math.gamma((1 + beta) / 2) * beta * math.pow(2, (beta - 1) / 2))
        xich_ma_1 = math.pow(up / down, 1 / beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def _shrink_encircling_Levy__(self, current_sea_lion, epoch_i, dist, c, beta=1):
        xich_ma_1, xich_ma_2 = self._caculate_xichma__(beta)
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (math.pow(np.abs(b), 1 / beta)) * dist * c
        D = np.random.uniform(self.domain_range[0], self.domain_range[1], 1)
        levy = LB * D
        return (current_sea_lion - math.sqrt(epoch_i + 1) * np.sign(np.random.random(1) - 0.5)) * levy

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        # Find prey which is the best solution
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)

        for i in range(self.epoch):

            for j in range(self.pop_size):

                SP_leader = np.random.uniform(0, 1)
                if SP_leader >= 0.5:
                    m = np.random.normal(0, 1)
                    new_position = np.abs(gbest[self.ID_POS] - pop[j][self.ID_POS]) * np.cos(2 * np.pi * m) \
                                        + gbest[self.ID_POS]
                else:
                    c = 2 - 2 * i / self.epoch
                    r1, r2 = np.random.normal(), np.random.normal()
                    b = np.random.uniform(0, 1, self.problem_size)
                    b1 = np.random.uniform()
                    b2 = np.random.uniform()

                    if c <= 1:
                        dist1 = b1 * np.abs(2 * gbest[self.ID_POS] - pop[j][self.ID_POS])
                        dist2 = b2 * np.abs(2 * pop[j][self.ID_PBEST_POS] - pop[j][self.ID_POS])
                        new_position = self._shrink_encircling_Levy__(pop[j][self.ID_POS], i, dist1, c)
                        # new_position = r1 * (gbest[self.ID_POS] - c * dist1) + \
                        #                r2 * (pop[j][self.ID_PBEST_POS] - c * dist2)
                    else:

                        random_SL_1 = np.random.uniform(self.domain_range[0], self.domain_range[1],
                                                        (self.problem_size, 1))
                        # random_SL_1 = gbest[self.ID_CURRENT_POSITION]

                        rand_index = np.random.randint(0, self.pop_size)
                        random_SL_2 = pop[rand_index][self.ID_PBEST_POS]
                        random_SL = random_SL_1 * 2 - random_SL_2

                        # dist = np.abs(b * random_SL - pop[j][self.ID_CURRENT_POSITION])
                        # new_position = random_SL - dist*c
                        new_position = random_SL
                # new_position[new_position < self.domain_range[0]] = self.domain_range[0]
                # new_position[new_position > self.domain_range[1]] = self.domain_range[1]

                new_position = self._evaluate_population__(new_position)
                new_fit = self._fitness_model__(solution=new_position, minmax=0)
                best_personal_fit = pop[j][self.ID_PBEST_FIT]

                # Cap nhat x hien tai, v hien tai, so sanh va cap nhat x past best voi x hien tai
                pop[j][self.ID_POS] = deepcopy(new_position)
                pop[j][self.ID_FIT] = new_fit

                if best_personal_fit >= new_fit:
                    pop[j][self.ID_PBEST_POS] = deepcopy(new_position)
                    pop[j][self.ID_PBEST_FIT] = new_fit

            current_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch = {}, Best fit so far = {}".format(i + 1, gbest[self.ID_FIT]))

        return gbest[self.ID_POS], self.loss_train
