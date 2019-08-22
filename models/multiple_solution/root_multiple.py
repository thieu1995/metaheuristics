import numpy as np
from copy import deepcopy

class RootAlgo(object):
    """ This is root of all Algorithms """
    ID_MIN_PROBLEM = 0
    ID_MAX_PROBLEM = -1

    def __init__(self, root_algo_paras = None):
        self.problem_size = root_algo_paras["problem_size"]
        self.domain_range = root_algo_paras["domain_range"]
        self.print_train = root_algo_paras["print_train"]
        self.objective_func = root_algo_paras["objective_func"]
        self.solution, self.loss_train = None, []

    def _create_solution__(self, minmax=0):
        solution = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution, minmax=minmax)
        return [solution, fitness]

    def _fitness_model__(self, solution=None, minmax=0):
        """ Assumption that objective function always return the original value """
        return self.objective_func(solution, self.problem_size) if minmax == 0 \
            else 1.0 / self.objective_func(solution, self.problem_size)

    def _fitness_encoded__(self, encoded=None, id_pos=None, minmax=0):
        return self._fitness_model__(solution=encoded[id_pos], minmax=minmax)

    def _get_global_best__(self, pop=None, id_fitness=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness])
        return deepcopy(sorted_pop[id_best])

    def _get_global_worst__(self, pop=None, id_fitness=None, id_worst=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness])
        return deepcopy(sorted_pop[id_worst])

    def _amend_solution__(self, solution=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0]:
                solution[i] = self.domain_range[0]
            if solution[i] > self.domain_range[1]:
                solution[i] = self.domain_range[1]

    def _amend_solution_and_return__(self, solution=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0]:
                solution[i] = self.domain_range[0]
            if solution[i] > self.domain_range[1]:
                solution[i] = self.domain_range[1]
        return solution

    ### This is failed version
    # def _amend_solution_and_return_failed__(self, solution=None):
    #     temp = deepcopy(solution)
    #     for i in range(self.problem_size):
    #         if solution[i] < self.domain_range[0]:
    #             solution[i] = self.domain_range[0]
    #         if solution[i] > self.domain_range[1]:
    #             solution[i] = self.domain_range[1]
    #     return temp


    def _create_opposition_solution__(self, solution=None, g_best=None):
        temp = [self.domain_range[0] + self.domain_range[1] - g_best[i] + np.random.random() * (g_best[i] - solution[i])
                      for i in range(self.problem_size)]
        return np.array(temp)


    def _train__(self):
        pass

