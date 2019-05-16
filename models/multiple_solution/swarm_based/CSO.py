import numpy as np
from random import sample, choice
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseCSO(RootAlgo):
    """
        This is basic version of Cat Swarm Optimization
        """
    # cat: x, v, fitness, flag
    ID_X = 0
    ID_V = 1
    ID_FIT = 2
    ID_FLAG = 3

    def __init__(self, root_algo_paras=None, cso_paras=None):
        """
        :param root_paras:
        :param cat_paras:
        # mixture_ratio - joining seeking mode with tracing mode
        # smp - seeking memory pool, 10 clones  (lon cang tot, nhugn ton time hon)
        # spc - self-position considering
        # cdc - counts of dimension to change  (lon cang tot)
        # srd - seeking range of the selected dimension (nho thi tot nhung search lau hon)
        # w_minmax - same in PSO
        # c1 - same in PSO
        # selected_strategy : 0: best fitness, 1: tournament, 2: roulette wheel, 3: random  (decrease by quality)
        """
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch =  cso_paras["epoch"]
        self.pop_size = cso_paras["pop_size"]
        self.mixture_ratio = cso_paras["mixture_ratio"]
        self.smp = cso_paras["smp"]
        self.spc = cso_paras["spc"]
        self.cdc = cso_paras["cdc"]
        self.srd = cso_paras["srd"]
        self.w_min = cso_paras["w_minmax"][0]
        self.w_max = cso_paras["w_minmax"][1]
        self.c1 = cso_paras["c1"]                    # Still using c1 and r1 but not c2, r2
        self.selected_strategy = cso_paras["selected_strategy"]

    def _create_solution__(self, minmax=0):
        """
                x: vi tri hien tai cua con meo
                v: vector van toc cua con meo (cung so chieu vs x)
                flag: trang thai cua meo, seeking hoac tracing
        """
        x = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        v = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(x, minmax)
        flag = False        # False: seeking mode , True: tracing mode
        if np.random.random() < self.mixture_ratio:
            flag = True
        return [x, v, fitness, flag]


    def _get_index_roulette_wheel_selection__(self, list_fitness, sum_fitness, fitness_min):
        r = np.random.uniform(fitness_min, sum_fitness)
        for idx, f in enumerate(list_fitness):
            r += f
            if r > sum_fitness:
                return idx

    def _seeking_mode__(self, cat):
        candidate_cats = []
        clone_cats = [deepcopy(cat) for _ in range(self.smp)]
        if self.spc:
            candidate_cats.append(deepcopy(cat))
            clone_cats = [deepcopy(cat) for _ in range(self.smp - 1)]

        for clone in clone_cats:
            idx = sample(range(0, self.problem_size), int(self.cdc * self.problem_size))
            for u in idx:
                if np.random.uniform() < 0.5:
                    # temp = clone[self.ID_X][u] * (1 + srd)
                    # if temp > 1.0:
                    #     clone[self.ID_X][u] = 1.0
                    clone[self.ID_X][u] += clone[self.ID_X][u] * self.srd
                else:
                    # temp = clone[self.ID_X][u] * (1 - srd)
                    # if temp < -1.0:
                    #     clone[self.ID_X][u] = -1.0
                    clone[self.ID_X][u] -= clone[self.ID_X][u] * self.srd
            clone[self.ID_FIT] = self._fitness_model__(clone[self.ID_X])
            candidate_cats.append(clone)

        fit1 = candidate_cats[0][self.ID_FIT]
        flag_equal = True
        for candidate in candidate_cats:
            if candidate[self.ID_FIT]!= fit1:
                flag_equal = False
                break

        if flag_equal == True:
            cat = choice(candidate_cats)            # Random choice one cat from list cats
        else:
            if self.selected_strategy == 0:                # Best fitness-self
                cat = sorted(candidate_cats, key=lambda cat: cat[self.ID_FIT])[0]

            elif self.selected_strategy == 1:              # Tournament
                k_way = 4
                idx = sample(range(0, self.smp), k_way)
                cats_k_way = [candidate_cats[_] for _ in idx]
                cat = sorted(cats_k_way, key=lambda cat: cat[self.ID_FIT])[0]

            elif self.selected_strategy == 2:              ### Roul-wheel selection
                fitness_list = [candidate_cats[u][self.ID_FIT] for u in range(0, len(candidate_cats))]
                fitness_sum = sum(fitness_list)
                fitness_min = min(fitness_list)
                idx = self._get_index_roulette_wheel_selection__(fitness_list, fitness_sum, fitness_min)
                cat = candidate_cats[idx]

            elif self.selected_strategy == 3:
                cat = choice(candidate_cats)                # Random
            else:
                print("Out of my abilities")
        return cat


    def _tracing_mode__(self, cat, cat_best, w):
        r1 = np.random.random()
        temp = w * cat[self.ID_V] + r1 * self.c1 * (cat_best[self.ID_X] - cat[self.ID_X])
        temp = np.where(temp > self.domain_range[1], self.domain_range[1], temp)
        cat[self.ID_X] += temp
        return cat

    def _train__(self):
        cats = [self._create_solution__() for _ in range(0, self.pop_size)]
        cat_best = self._get_global_best__(cats, self.ID_FIT, self.ID_MIN_PROBLEM)

        for i in range(self.epoch):
            w = (self.epoch - i) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for q in range(0, self.pop_size):
                if cats[q][self.ID_FLAG]:
                    cats[q] = self._tracing_mode__(cats[q], cat_best, w)
                else:
                    cats[q] = self._seeking_mode__(cats[q])
            for q in range(0, self.pop_size):
                if np.random.uniform() < self.mixture_ratio:
                    cats[q][self.ID_FLAG] = True
                else:
                    cats[q][self.ID_FLAG] = False

            cat_current_best = self._get_global_best__(cats, self.ID_FIT, self.ID_MIN_PROBLEM)

            if cat_best[self.ID_FIT]> cat_current_best[self.ID_FIT]:
                cat_best = cat_current_best
            self.loss_train.append(cat_best[self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, average MAE over population: {1}".format(i+1, cat_best[self.ID_FIT]))

        return cat_best[self.ID_X], self.loss_train




