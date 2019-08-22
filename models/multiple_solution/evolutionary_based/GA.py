import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BaseGA(RootAlgo):
    """
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """
    ID_FITNESS = 1

    def __init__(self, root_algo_paras=None, ga_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = ga_paras["epoch"]
        self.pop_size = ga_paras["pop_size"]
        self.pc = ga_paras["pc"]
        self.pm = ga_paras["pm"]

    ### Selection
    def _get_index_roulette_wheel_selection__(self, list_fitness, sum_fitness):
        r = np.random.uniform(low=0, high=sum_fitness)
        for idx, f in enumerate(list_fitness):
            r = r + f
            if r > sum_fitness:
                return idx

    ### Crossover
    def _crossover_arthmetic_recombination__(self, dad=None, mom=None):
        r = np.random.uniform()             # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad) + np.multiply((1 - r), mom)
        w2 = np.multiply(r, mom) + np.multiply((1 - r), dad)
        return w1, w2

    ### Mutation
    def _mutation_flip_point__(self, parent, index):
        w = deepcopy(parent)
        w[index] = np.random.uniform(self.domain_range[0], self.domain_range[1])
        return w

    def _create_next_generation__(self, pop):
        next_population = []

        list_fitness = [pop[i][self.ID_FITNESS] for i in range(self.pop_size)]
        fitness_sum = sum(list_fitness)
        while (len(next_population) < self.pop_size):
            ### Selection
            c1 = deepcopy( pop[self._get_index_roulette_wheel_selection__(list_fitness, fitness_sum)] )
            c2 = deepcopy( pop[self._get_index_roulette_wheel_selection__(list_fitness, fitness_sum)] )

            w1, w2 = deepcopy(c1[0]), deepcopy(c2[0])
            ### Crossover
            if np.random.uniform() < self.pc:
                w1, w2 = self._crossover_arthmetic_recombination__(c1[0], c2[0])

            ### Mutation
            for id in range(0, self.problem_size):
                if np.random.uniform() < self.pm:
                    w1 = self._mutation_flip_point__(w1, id)
                if np.random.uniform() < self.pm:
                    w2 = self._mutation_flip_point__(w2, id)

            c1_new = [deepcopy(w1), self._fitness_model__(w1, minmax=1)]
            c2_new = [deepcopy(w2), self._fitness_model__(w2, minmax=1)]
            next_population.append(c1_new)
            next_population.append(c2_new)
        return next_population


    def _train__(self):
        best_train = [None, -1 ]
        pop = [self._create_solution__(minmax=1) for _ in range(self.pop_size)]

        for j in range(0, self.epoch):
            # Next generations
            pop = deepcopy(self._create_next_generation__(pop))
            current_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FITNESS, id_best=self.ID_MAX_PROBLEM)
            if current_best[self.ID_FITNESS] > best_train[self.ID_FITNESS]:
                best_train = current_best
            if self.print_train:
                print("> Epoch {0}: Best training fitness {1}".format(j + 1, 1.0 / best_train[self.ID_FITNESS]))
            self.loss_train.append(np.power(best_train[self.ID_FITNESS], -1))

        return best_train[0], self.loss_train
