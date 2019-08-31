import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseDE(RootAlgo):
    """
    Differential Evolution : Taken from here
    http://www.cleveralgorithms.com/nature-inspired/evolution/differential_evolution.html
    """
    ID_SOl = 0
    ID_FIT = 1
    def __init__(self, root_algo_paras=None, de_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch =  de_paras["epoch"]
        self.pop_size = de_paras["pop_size"]
        self.weighting_factor = de_paras["Wf"]
        self.crossover_rate = de_paras["Cr"]

    def _mutation__(self, p0, p1, p2, p3):
        # Choose a cut point which differs 0 and chromosome-1 (first and last element)
        cut_point = np.random.randint(1, self.problem_size - 1)
        sample = []
        for i in range(self.problem_size):
            if i == cut_point or np.random.uniform() < self.crossover_rate  :
                v = p1[i] + self.weighting_factor * ( p2[i] - p3[i])
                v = self.domain_range[0] if v < self.domain_range[0] else v
                v = self.domain_range[1] if v > self.domain_range[1] else v
                sample.append(v)
            else :
                sample.append(p0[i])
        return np.array(sample)

    def _create_children__(self, pop):
        new_children = []
        for i in range(self.pop_size):
            temp = np.random.choice(range(0, self.pop_size), 3, replace=False)
            while i in temp:
                temp = np.random.choice(range(0, self.pop_size), 3, replace=False)
            #create new child and append in children array
            child = self._mutation__(pop[i][self.ID_SOl], pop[temp[0]][self.ID_SOl], pop[temp[1]][self.ID_SOl], pop[temp[2]][self.ID_SOl])
            fit = self._fitness_model__(child)
            new_children.append([child, fit])
        return np.array(new_children)

    ### Survivor Selection
    def _greedy_selection__(self, pop_old=None, pop_new=None):
        pop = [pop_new[i] if pop_new[i][self.ID_FIT] < pop_old[i][self.ID_FIT]
               else pop_old[i] for i in range(self.pop_size)]
        return pop

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)

        for i in range(self.epoch):
            # create children
            children = self._create_children__(pop)
            # create new pop by comparing fitness of corresponding each member in pop and children
            pop = self._greedy_selection__(pop, children)

            current_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT]< gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_FIT])
            if self.print_train:
                print("Epoch : {}, [MSE, MAE]: {}".format(i + 1, gbest[self.ID_FIT]))

        return gbest[self.ID_SOl], self.loss_train





