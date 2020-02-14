import numpy as np
from copy import deepcopy
import math
from models.multiple_solution.root_multiple import RootAlgo


class BaseGSO(RootAlgo):
    """
    Galactic Swarm Optimization
    """
    ID_CURRENT_POS = 0
    ID_BEST_PAST_POS = 1
    ID_VECTOR_V = 2
    ID_CURRENT_FIT = 3
    ID_PAST_FIT = 4

    def __init__(self, root_algo_paras=None, gso_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = gso_paras["epoch"][0]
        self.epoch_subswarm = gso_paras["epoch"][1]
        self.epoch_superswarm = gso_paras["epoch"][2]
        self.pop_size = gso_paras["pop_size"]
        self.c1 = gso_paras["c_minmax"][0]
        self.c2 = gso_paras["c_minmax"][1]
        self.w_min = gso_paras["w_minmax"][0]
        self.w_max = gso_paras["w_minmax"][1]
        self.num_subswarm = gso_paras["num_subswarm"]

    def _create_solution__(self, minmax=0):
        """  This algorithm has different encoding mechanism, so we need to override this method
                x: current position
                x_past_best: the best personal position so far (in history)
                v: velocity of this bird (same number of dimension of x)
        """
        x = np.random.uniform(self.domain_range[0], self.domain_range[1], (self.problem_size, 1))
        x_past_best = deepcopy(x)
        v = np.zeros((len(x), 1))
        x_fitness = self._fitness_model__(solution=x, minmax=minmax)
        x_past_fitness = deepcopy(x_fitness)
        return [x, x_past_best, v, x_fitness, x_past_fitness]

    def _create_subswarm_collection(self, pop):
        num_particle_per_subswarm = int(self.pop_size/self.num_subswarm)
        subswarm_collection = []
        for i in range(self.num_subswarm):
            start_idx = num_particle_per_subswarm*i
            end_idx = num_particle_per_subswarm*(i+1)
            subswarm_i = pop[start_idx:end_idx]
            subswarm_collection.append(subswarm_i)
        return subswarm_collection

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
        subswarm_collection = self._create_subswarm_collection(pop)

        for iter in range(self.epoch):
            superswarm = []
            for x in range(len(subswarm_collection)):
                subswarm_i = subswarm_collection[x]
                gbest_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
                for i in range(self.epoch_subswarm):
                    w = (self.epoch_subswarm - i) / self.epoch_subswarm * (self.w_max - self.w_min) + self.w_min
                    for j in range(int(self.pop_size/self.num_subswarm)):
                        r1 = np.random.random_sample()
                        r2 = np.random.random_sample()
                        vi_sau = w * subswarm_i[j][self.ID_VECTOR_V] + self.c1 * r1 * \
                                 (subswarm_i[j][self.ID_BEST_PAST_POS] - subswarm_i[j][self.ID_CURRENT_POS]) \
                                 + self.c2 * r2 * (gbest[self.ID_CURRENT_POS] - subswarm_i[j][self.ID_CURRENT_POS])
                        vi_sau = np.maximum(vi_sau, -0.1 * 10)
                        vi_sau = np.minimum(vi_sau, 0.1 * 10)
                        xi_sau = subswarm_i[j][self.ID_CURRENT_POS] + vi_sau  # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                        fit_sau = self._fitness_model__(solution=xi_sau, minmax=0)
                        fit_truoc = subswarm_i[j][self.ID_PAST_FIT]

                        # Update current position, current vector v
                        subswarm_i[j][self.ID_CURRENT_POS] = deepcopy(xi_sau)
                        subswarm_i[j][self.ID_VECTOR_V] = deepcopy(vi_sau)
                        subswarm_i[j][self.ID_CURRENT_FIT] = fit_sau

                        if fit_sau < fit_truoc:
                            subswarm_i[j][self.ID_BEST_PAST_POS] = deepcopy(xi_sau)
                            subswarm_i[j][self.ID_PAST_FIT] = fit_sau

                    current_best_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT,
                                                           id_best=self.ID_MIN_PROBLEM)
                    if current_best_i[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                        gbest_i = deepcopy(current_best_i)
                superswarm.append(gbest_i)
                subswarm_collection[x] = deepcopy(subswarm_i)

            for i in range(self.epoch_superswarm):
                # Update weight after each move count  (weight down)
                w = (self.epoch_superswarm - i) / self.epoch_superswarm * (self.w_max - self.w_min) + self.w_min
                for j in range(self.num_subswarm):
                    r1 = np.random.random_sample()
                    r2 = np.random.random_sample()
                    vi_sau = w * superswarm[j][self.ID_VECTOR_V] + self.c1 * r1 * \
                             (superswarm[j][self.ID_BEST_PAST_POS] - superswarm[j][self.ID_CURRENT_POS]) \
                             + self.c2 * r2 * (gbest[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                    vi_sau = np.maximum(vi_sau, -0.1 * 10)
                    vi_sau = np.minimum(vi_sau, 0.1 * 10)
                    xi_sau = superswarm[j][self.ID_CURRENT_POS] + vi_sau  # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                    fit_sau = self._fitness_model__(solution=xi_sau, minmax=0)
                    fit_truoc = superswarm[j][self.ID_PAST_FIT]

                    # Update current position, current vector v
                    superswarm[j][self.ID_CURRENT_POS] = deepcopy(xi_sau)
                    superswarm[j][self.ID_VECTOR_V] = deepcopy(vi_sau)
                    superswarm[j][self.ID_CURRENT_FIT] = fit_sau

                    if fit_sau < fit_truoc:
                        superswarm[j][self.ID_BEST_PAST_POS] = deepcopy(xi_sau)
                        superswarm[j][self.ID_PAST_FIT] = fit_sau

                current_best = self._get_global_best__(pop=superswarm, id_fitness=self.ID_CURRENT_FIT,
                                                       id_best=self.ID_MIN_PROBLEM)
                if current_best[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                    gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_CURRENT_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(iter+1, gbest[self.ID_CURRENT_FIT]))

        return gbest[self.ID_CURRENT_POS], self.loss_train


class GSOWOA(BaseGSO):
    """
    Galactic Swarm Optimization
    """
    ID_CURRENT_POS = 0
    ID_BEST_PAST_POS = 1
    ID_VECTOR_V = 2
    ID_CURRENT_FIT = 3
    ID_PAST_FIT = 4

    def __init__(self, root_algo_paras=None, gso_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = gso_paras["epoch"][0]
        self.epoch_subswarm = gso_paras["epoch"][1]
        self.epoch_superswarm = gso_paras["epoch"][2]
        self.pop_size = gso_paras["pop_size"]
        self.c1 = gso_paras["c_minmax"][0]
        self.c2 = gso_paras["c_minmax"][1]
        self.w_min = gso_paras["w_minmax"][0]
        self.w_max = gso_paras["w_minmax"][1]
        self.num_subswarm = gso_paras["num_subswarm"]

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
        subswarm_collection = self._create_subswarm_collection(pop)

        for iter in range(self.epoch):
            superswarm = []
            for x in range(len(subswarm_collection)):
                subswarm_i = subswarm_collection[x]
                gbest_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
                for i in range(self.epoch_subswarm):
                    w = (self.epoch_subswarm - i) / self.epoch_subswarm * (self.w_max - self.w_min) + self.w_min
                    for j in range(int(self.pop_size/self.num_subswarm)):
                        r1 = np.random.random_sample()
                        r2 = np.random.random_sample()
                        vi_sau = w * subswarm_i[j][self.ID_VECTOR_V] + self.c1 * r1 * \
                                 (subswarm_i[j][self.ID_BEST_PAST_POS] - subswarm_i[j][self.ID_CURRENT_POS]) \
                                 + self.c2 * r2 * (gbest[self.ID_CURRENT_POS] - subswarm_i[j][self.ID_CURRENT_POS])
                        vi_sau = np.maximum(vi_sau, -0.1 * 10)
                        vi_sau = np.minimum(vi_sau, 0.1 * 10)
                        xi_sau = subswarm_i[j][self.ID_CURRENT_POS] + vi_sau  # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                        fit_sau = self._fitness_model__(solution=xi_sau, minmax=0)
                        fit_truoc = subswarm_i[j][self.ID_PAST_FIT]

                        # Update current position, current vector v
                        subswarm_i[j][self.ID_CURRENT_POS] = deepcopy(xi_sau)
                        subswarm_i[j][self.ID_VECTOR_V] = deepcopy(vi_sau)
                        subswarm_i[j][self.ID_CURRENT_FIT] = fit_sau

                        if fit_sau < fit_truoc:
                            subswarm_i[j][self.ID_BEST_PAST_POS] = deepcopy(xi_sau)
                            subswarm_i[j][self.ID_PAST_FIT] = fit_sau

                    current_best_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT,
                                                           id_best=self.ID_MIN_PROBLEM)
                    if current_best_i[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                        gbest_i = deepcopy(current_best_i)
                superswarm.append(gbest_i)
                subswarm_collection[x] = deepcopy(subswarm_i)

            for i in range(self.epoch_superswarm):
                a = 2 - 2 * i / (self.epoch - 1)  # linearly decreased from 2 to 0

                for j in range(self.num_subswarm):

                    r = np.random.rand()
                    A = 2 * a * r - a
                    C = 2 * r
                    l = np.random.uniform(-1, 1)
                    p = np.random.rand()
                    b = 1

                    if p < 0.5:
                        if np.abs(A) < 1:
                            D = np.abs(C * gbest[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                            new_position = gbest[self.ID_CURRENT_POS] - A * D
                        else:
                            x_rand = superswarm[np.random.randint(self.num_subswarm)]  # chon ra 1 thang random
                            D = np.abs(C * x_rand[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                            new_position = (x_rand[self.ID_CURRENT_POS] - A * D)
                    else:
                        D1 = np.abs(gbest[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                        new_position = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest[self.ID_CURRENT_POS]
                    new_position[new_position < self.domain_range[0]] = self.domain_range[0]
                    new_position[new_position > self.domain_range[1]] = self.domain_range[1]
                    fit = self._fitness_model__(new_position)
                    superswarm[j][self.ID_CURRENT_POS] = new_position
                    superswarm[j][self.ID_CURRENT_FIT] = fit

                current_best = self._get_global_best__(pop=superswarm, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
                if current_best[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                    gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_CURRENT_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(iter+1, gbest[self.ID_CURRENT_FIT]))

        return gbest[self.ID_CURRENT_POS], self.loss_train


class GSOLWOA(GSOWOA):
    """
    Galactic Swarm Optimization
    """
    ID_CURRENT_POS = 0
    ID_BEST_PAST_POS = 1
    ID_VECTOR_V = 2
    ID_CURRENT_FIT = 3
    ID_PAST_FIT = 4

    def __init__(self, root_algo_paras=None, gso_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = gso_paras["epoch"][0]
        self.epoch_subswarm = gso_paras["epoch"][1]
        self.epoch_superswarm = gso_paras["epoch"][2]
        self.pop_size = gso_paras["pop_size"]
        self.c1 = gso_paras["c_minmax"][0]
        self.c2 = gso_paras["c_minmax"][1]
        self.w_min = gso_paras["w_minmax"][0]
        self.w_max = gso_paras["w_minmax"][1]
        self.num_subswarm = gso_paras["num_subswarm"]

    def caculate_xichma(self, beta):
        up = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        down = (math.gamma((1 + beta) / 2) * beta * math.pow(2, (beta - 1) / 2))
        xich_ma_1 = math.pow(up / down, 1 / beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def shrink_encircling_Levy(self, current_whale, best_solution, epoch_i, C, beta=1):
        xich_ma_1, xich_ma_2 = self.caculate_xichma(beta)
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (math.pow(np.abs(b), 1 / beta)) * (-1 * current_whale + C * best_solution)
        D = np.random.uniform(self.domain_range[0], self.domain_range[1], 1)
        levy = LB * D
        return (current_whale + 1/math.sqrt(epoch_i + 1)) * levy

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
        subswarm_collection = self._create_subswarm_collection(pop)

        for iter in range(self.epoch):
            superswarm = []
            for x in range(len(subswarm_collection)):
                subswarm_i = subswarm_collection[x]
                gbest_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
                for i in range(self.epoch_subswarm):
                    w = (self.epoch_subswarm - i) / self.epoch_subswarm * (self.w_max - self.w_min) + self.w_min
                    for j in range(int(self.pop_size/self.num_subswarm)):
                        r1 = np.random.random_sample()
                        r2 = np.random.random_sample()
                        vi_sau = w * subswarm_i[j][self.ID_VECTOR_V] + self.c1 * r1 * \
                                 (subswarm_i[j][self.ID_BEST_PAST_POS] - subswarm_i[j][self.ID_CURRENT_POS]) \
                                 + self.c2 * r2 * (gbest[self.ID_CURRENT_POS] - subswarm_i[j][self.ID_CURRENT_POS])
                        vi_sau = np.maximum(vi_sau, -0.1 * 10)
                        vi_sau = np.minimum(vi_sau, 0.1 * 10)
                        xi_sau = subswarm_i[j][self.ID_CURRENT_POS] + vi_sau  # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                        fit_sau = self._fitness_model__(solution=xi_sau, minmax=0)
                        fit_truoc = subswarm_i[j][self.ID_PAST_FIT]

                        # Update current position, current vector v
                        subswarm_i[j][self.ID_CURRENT_POS] = deepcopy(xi_sau)
                        subswarm_i[j][self.ID_VECTOR_V] = deepcopy(vi_sau)
                        subswarm_i[j][self.ID_CURRENT_FIT] = fit_sau

                        if fit_sau < fit_truoc:
                            subswarm_i[j][self.ID_BEST_PAST_POS] = deepcopy(xi_sau)
                            subswarm_i[j][self.ID_PAST_FIT] = fit_sau

                    current_best_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT,
                                                           id_best=self.ID_MIN_PROBLEM)
                    if current_best_i[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                        gbest_i = deepcopy(current_best_i)
                superswarm.append(gbest_i)
                subswarm_collection[x] = deepcopy(subswarm_i)

            for i in range(self.epoch_superswarm):
                a = 2 - 2 * i / (self.epoch - 1)  # linearly decreased from 2 to 0

                for j in range(self.num_subswarm):

                    r = np.random.rand()
                    A = 2 * a * r - a
                    C = 2 * r
                    l = np.random.uniform(-1, 1)
                    p = np.random.rand()
                    b = 1

                    if p < 0.5:
                        if np.abs(A) < 1:
                            new_position = self.shrink_encircling_Levy(superswarm[j][self.ID_CURRENT_POS],
                                                                       gbest[self.ID_CURRENT_POS],
                                                                       i,
                                                                       C)
                        else:
                            x_rand = superswarm[np.random.randint(self.num_subswarm)]  # chon ra 1 thang random
                            D = np.abs(C * x_rand[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                            new_position = (x_rand[self.ID_CURRENT_POS] - A * D)
                    else:
                        D1 = np.abs(gbest[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                        new_position = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest[self.ID_CURRENT_POS]
                    new_position = self._amend_solution_and_return__(new_position)
                    fit = self._fitness_model__(new_position)
                    superswarm[j][self.ID_CURRENT_POS] = new_position
                    superswarm[j][self.ID_CURRENT_FIT] = fit

                current_best = self._get_global_best__(pop=superswarm, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
                if current_best[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                    gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_CURRENT_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(iter+1, gbest[self.ID_CURRENT_FIT]))

        return gbest[self.ID_CURRENT_POS], self.loss_train


class HGEW(GSOLWOA):
    """
    Galactic Swarm Optimization
    """
    ID_CURRENT_POS = 0
    ID_BEST_PAST_POS = 1
    ID_VECTOR_V = 2
    ID_CURRENT_FIT = 3
    ID_PAST_FIT = 4

    def __init__(self, root_algo_paras=None, gso_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = gso_paras["epoch"][0]
        self.epoch_subswarm = gso_paras["epoch"][1]
        self.epoch_superswarm = gso_paras["epoch"][2]
        self.pop_size = gso_paras["pop_size"]
        self.c1 = gso_paras["c_minmax"][0]
        self.c2 = gso_paras["c_minmax"][1]
        self.w_min = gso_paras["w_minmax"][0]
        self.w_max = gso_paras["w_minmax"][1]
        self.num_subswarm = gso_paras["num_subswarm"]

    def crossover(self, superswarm, gbest):
        partner_index = np.random.randint(0, self.num_subswarm)
        partner = superswarm[partner_index][self.ID_CURRENT_POS]
        # partner = np.random.uniform(self.range0, self.range1, self.dimension)

        start_point = np.random.randint(0, self.problem_size / 2)
        new_whale = np.zeros((self.problem_size, 1))

        index1 = start_point
        index2 = int(start_point + self.problem_size / 2)
        index3 = int(self.problem_size)

        new_whale[0:index1] = gbest[self.ID_CURRENT_POS][0:index1]
        new_whale[index1:index2] = partner[index1:index2]
        new_whale[index2:index3] = gbest[self.ID_CURRENT_POS][index2:index3]

        return new_whale

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
        subswarm_collection = self._create_subswarm_collection(pop)

        for iter in range(self.epoch):
            superswarm = []
            for x in range(len(subswarm_collection)):
                subswarm_i = subswarm_collection[x]
                gbest_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
                for i in range(self.epoch_subswarm):
                    w = (self.epoch_subswarm - i) / self.epoch_subswarm * (self.w_max - self.w_min) + self.w_min
                    for j in range(int(self.pop_size/self.num_subswarm)):
                        r1 = np.random.random_sample()
                        r2 = np.random.random_sample()
                        vi_sau = w * subswarm_i[j][self.ID_VECTOR_V] + self.c1 * r1 * \
                                 (subswarm_i[j][self.ID_BEST_PAST_POS] - subswarm_i[j][self.ID_CURRENT_POS]) \
                                 + self.c2 * r2 * (gbest[self.ID_CURRENT_POS] - subswarm_i[j][self.ID_CURRENT_POS])
                        vi_sau = np.maximum(vi_sau, -0.1 * 10)
                        vi_sau = np.minimum(vi_sau, 0.1 * 10)
                        xi_sau = subswarm_i[j][self.ID_CURRENT_POS] + vi_sau  # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                        fit_sau = self._fitness_model__(solution=xi_sau, minmax=0)
                        fit_truoc = subswarm_i[j][self.ID_PAST_FIT]

                        # Update current position, current vector v
                        subswarm_i[j][self.ID_CURRENT_POS] = deepcopy(xi_sau)
                        subswarm_i[j][self.ID_VECTOR_V] = deepcopy(vi_sau)
                        subswarm_i[j][self.ID_CURRENT_FIT] = fit_sau

                        if fit_sau < fit_truoc:
                            subswarm_i[j][self.ID_BEST_PAST_POS] = deepcopy(xi_sau)
                            subswarm_i[j][self.ID_PAST_FIT] = fit_sau

                    current_best_i = self._get_global_best__(pop=subswarm_i, id_fitness=self.ID_CURRENT_FIT,
                                                           id_best=self.ID_MIN_PROBLEM)
                    if current_best_i[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                        gbest_i = deepcopy(current_best_i)
                superswarm.append(gbest_i)
                subswarm_collection[x] = deepcopy(subswarm_i)

            for i in range(self.epoch_superswarm):
                a = 2 - 2 * i / (self.epoch - 1)  # linearly decreased from 2 to 0

                for j in range(self.num_subswarm):

                    r = np.random.rand()
                    A = 2 * a * r - a
                    C = 2 * r
                    l = np.random.uniform(-1, 1)
                    p = np.random.rand()
                    p1 = np.random.random()
                    b = 1

                    if p < 0.5:
                        if np.abs(A) < 1:
                            new_position = self.shrink_encircling_Levy(superswarm[j][self.ID_CURRENT_POS],
                                                                       gbest[self.ID_CURRENT_POS],
                                                                       i,
                                                                       C)
                        else:
                            if p1 < 0.6:
                                x_rand = superswarm[np.random.randint(self.num_subswarm)]
                            else:
                                x_rand = self.crossover(superswarm, gbest)
                            D = np.abs(C * x_rand[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                            new_position = (x_rand[self.ID_CURRENT_POS] - A * D)
                    else:
                        D1 = np.abs(gbest[self.ID_CURRENT_POS] - superswarm[j][self.ID_CURRENT_POS])
                        new_position = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest[self.ID_CURRENT_POS]
                    new_position = self._amend_solution_and_return__(new_position)
                    fit = self._fitness_model__(new_position)
                    superswarm[j][self.ID_CURRENT_POS] = new_position
                    superswarm[j][self.ID_CURRENT_FIT] = fit

                current_best = self._get_global_best__(pop=superswarm, id_fitness=self.ID_CURRENT_FIT, id_best=self.ID_MIN_PROBLEM)
                if current_best[self.ID_CURRENT_FIT] < gbest[self.ID_CURRENT_FIT]:
                    gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_CURRENT_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(iter+1, gbest[self.ID_CURRENT_FIT]))

        return gbest[self.ID_CURRENT_POS], self.loss_train