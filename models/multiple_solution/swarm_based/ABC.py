from random import random, randint
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseABC(RootAlgo):
    """
    - Transformation from ruby code to python code in Clever Algorithms book
        - Improved function _create_neigh_bee__
        - Better results, faster convergence
    """
    ID_BEE = 0
    ID_FITNESS = 1

    def __init__(self, root_algo_paras=None, abc_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = abc_paras["epoch"]
        self.pop_size = abc_paras["pop_size"]
        self.e_bees = abc_paras["couple_bees"][0]
        self.o_bees = abc_paras["couple_bees"][1]

        self.patch_size = abc_paras["patch_variables"][0]
        self.patch_factor = abc_paras["patch_variables"][1]
        self.num_sites = abc_paras["sites"][0]
        self.elite_sites = abc_paras["sites"][1]

    def _create_neigh_bee__(self, individual=None, patch_size=None):
        t1 = randint(0, len(individual) - 1)
        new_bee = deepcopy(individual)
        new_bee[t1] = (individual[t1] + random() * patch_size) if random() < 0.5 else (individual[t1] - random() * patch_size)
        if random() < 0.5:
            new_bee[t1] = individual[t1] + random() * patch_size
            if new_bee[t1] > self.domain_range[1]:
                new_bee[t1] = self.domain_range[1]
        else:
            new_bee[t1] = individual[t1] - random() * patch_size
            if new_bee[t1] < self.domain_range[0]:
                new_bee[t1] = self.domain_range[0]
        return [new_bee, self._fitness_model__(new_bee)]


    def _search_neigh__(self, parent=None, neigh_size=None):  # parent:  [ vector_individual, fitness ]
        """
        Seeking in neigh_size neighborhood, take the best
        """
        neigh = [self._create_neigh_bee__(parent[0], self.patch_size) for _ in range(0, neigh_size)]
        return self._get_global_best__(neigh, self.ID_FITNESS, self.ID_MIN_PROBLEM)

    def _create_scout_bees__(self, num_scouts=None):  # So luong ong trinh tham
        return [self._create_solution__() for _ in range(0, num_scouts)]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        for j in range(0, self.epoch):
            pop_sorted = sorted(pop, key=lambda bee: bee[self.ID_FITNESS])
            best = deepcopy(pop_sorted[self.ID_MIN_PROBLEM])

            next_gen = []
            for i in range(0, self.num_sites):
                if i < self.elite_sites:
                    neigh_size = self.e_bees
                else:
                    neigh_size = self.o_bees
                next_gen.append(self._search_neigh__(pop_sorted[i], neigh_size))

            scouts = self._create_scout_bees__(self.pop_size - self.num_sites)  # Ong trinh tham
            pop = next_gen + scouts
            self.patch_size = self.patch_size * self.patch_factor
            self.loss_train.append(best[self.ID_FITNESS])
            if self.print_train:
                print("Epoch = {}, patch_size = {}, Fit = {}".format(j + 1, self.patch_size, best[self.ID_FITNESS]))

        best = self._get_global_best__(pop, self.ID_FITNESS, self.ID_MIN_PROBLEM)
        return best[self.ID_BEE], self.loss_train




