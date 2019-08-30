import numpy as np
from numpy.random import uniform
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseBOA(RootAlgo):
    """
    This is the version I implemented as the paper said:
        Butterfly optimization algorithm: a novel approach for global optimization
    Really cant even converge. Totally bullshit
    """
    ID_POS = 0
    ID_FIT = 1
    ID_FRA = 2

    def __init__(self, root_algo_paras=None, boa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = boa_paras["epoch"]
        self.pop_size = boa_paras["pop_size"]
        self.c = boa_paras["c"]                 # 0.01, is the sensory modality
        self.p = boa_paras["p"]                 # 0.8, Search for food and mating partner by butterflies can occur at both local and global scale
        self.alpha = boa_paras["alpha"]         # 0.1-0.3 (0 -> vo cung), the power exponent dependent on modality

    def _create_solution__(self, minmax=0):
        x = np.random.uniform(self.domain_range[0], self.domain_range[1], (self.problem_size, 1))
        fit = self._fitness_model__(solution=x, minmax=minmax)      # stimulus intensity Ii
        fragrance = 0
        return [x, fit, fragrance]

    def _train__(self):
        alpha = self.alpha[0]
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                pop[i][self.ID_FRA] = self.c * pop[i][self.ID_FIT]**alpha


            for i in range(self.pop_size):
                r = np.random.uniform()
                t1 = None
                if r < self.p:
                    t1 = pop[i][self.ID_POS] + (r**2 * g_best[self.ID_POS] - pop[i][self.ID_POS]) * pop[i][self.ID_FRA]
                else:
                    idx = np.random.randint(0, self.pop_size)
                    t1 = pop[i][self.ID_POS] + (r**2 * pop[idx][self.ID_POS] - pop[i][self.ID_POS]) * pop[i][self.ID_FRA]
                fit = self._fitness_model__(t1, self.ID_MIN_PROBLEM)
                fra = self.c * fit**alpha
                pop[i] = [t1, fit, fra]
            alpha = self.alpha[0] + ((epoch +1)/self.epoch) * (self.alpha[1] - self.alpha[0])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], self.loss_train




class OriginalBOA(RootAlgo):
    """
    Original version of Butterfly Optimization Algorithm:
        Butterfly optimization algorithm: a novel approach for global optimization - 2019

    This is the code of the original author of BOA. He public on mathworks. But take a look at his code and his paper.
    That is completely different. I don't want to say it is trash, but really?

    Let's talk about the results.
    If I use his mathworks, the results is good for single-global-optima such as square_funtion (whale_f1, whale_f2,...)
        But really bad, It can't even converge when dealing with multiple-global-optima such as CEC2014 (C1, C2, ...)
    If I use the version I coded based on his paper. It can't converge.

    https://www.mathworks.com/matlabcentral/fileexchange/68209-butterfly-optimization-algorithm-boa

    So many people asking him public the code of function, which used in the paper. Even 1 guy said
        "Honestly,this algorithm looks like Flower Pollination Algorithm developed by Yang."
    """

    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, boa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = boa_paras["epoch"]
        self.pop_size = boa_paras["pop_size"]
        self.c = boa_paras["c"]                 # 0.01, is the sensory modality
        self.p = boa_paras["p"]                 # 0.8, Search for food and mating partner by butterflies can occur at both local and global scale
        self.alpha = boa_paras["alpha"]         # 0.1-0.3 (0 -> vo cung), the power exponent dependent on modality

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
        c_temp = self.c
        alpha = self.alpha[0]

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                FP = c_temp * (pop[i][self.ID_FIT] ** alpha)

                t1 = None
                if uniform() < self.p:
                    t1 = pop[i][self.ID_POS] + (uniform()*uniform() * g_best[self.ID_POS] - pop[i][self.ID_POS]) * FP
                else:
                    epsilon = uniform()
                    id1, id2 = np.random.randint(0, self.pop_size, 2)
                    dis = (epsilon**2) * pop[id1][self.ID_POS] - pop[id2][self.ID_POS]
                    t1 = pop[i][self.ID_POS] + dis * FP
                t1 = self._amend_solution_and_return__(t1)
                fit = self._fitness_model__(t1)

                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [deepcopy(t1), fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [deepcopy(t1), fit]

            c_temp = c_temp + 0.025 / (c_temp * self.epoch)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], self.loss_train


class AdaptiveBOA(RootAlgo):
    """
    A variant version of Butterfly Optimization Algorithm:
        A novel adaptive butterfly optimization algorithm - 2018
        https://sci-hub.tw/10.1142/s2047684118500264

    Wow, my mind just blown up when I found out that this guy:
        https://scholar.google.co.in/citations?hl=en&user=KvcHovcAAAAJ&view_op=list_works&sortby=pubdate
    He invent BOA algorithm and public it in 2019, but so many variant version of BOA has been created since 2015. How
        the hell that happened?
    This is a plagiarism? I think this is one of the most biggest reason why mathematic researchers calling out
        meta-heuristics community is completely bullshit and unethical.
    Just for producing more trash paper without any knowledge in it? This is why I listed BOA as the totally trash.

    """

    ID_POS = 0
    ID_FIT = 1

    def __init__(self, root_algo_paras=None, boa_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = boa_paras["epoch"]
        self.pop_size = boa_paras["pop_size"]
        self.c = boa_paras["c"]                 # 0.01, is the sensory modality
        self.p = boa_paras["p"]                 # 0.8, Search for food and mating partner by butterflies can occur at both local and global scale
        self.alpha = boa_paras["alpha"]         # 0.1-0.3 (0 -> vo cung), the power exponent dependent on modality

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
        c_temp = self.c
        alpha = self.alpha[0]

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                FP = c_temp * (pop[i][self.ID_FIT] ** alpha)

                t1 = None
                if uniform() < self.p:
                    t1 = pop[i][self.ID_POS] + (uniform()*uniform() * g_best[self.ID_POS] - pop[i][self.ID_POS]) * FP
                else:
                    epsilon = uniform()
                    id1, id2 = np.random.randint(0, self.pop_size, 2)
                    dis = (epsilon**2) * pop[id1][self.ID_POS] - pop[id2][self.ID_POS]
                    t1 = pop[i][self.ID_POS] + dis * FP
                t1 = self._amend_solution_and_return__(t1)
                fit = self._fitness_model__(t1)

                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [deepcopy(t1), fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [deepcopy(t1), fit]

            c_temp = c_temp * (10.0**(-5)/0.9)**(2.0 / self.epoch)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], self.loss_train



