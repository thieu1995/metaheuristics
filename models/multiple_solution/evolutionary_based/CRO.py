import numpy as np
import random
from models.multiple_solution.root_multiple import RootAlgo

class BaseCRO(RootAlgo):
    """
    This is standard version of CRO implement according to this paper:
        http://downloads.hindawi.com/journals/tswj/2014/739768.pdf
    """
    HEALTH = 1000000
    def __init__(self, root_algo_paras=None, cro_paras = None):
        """
        # reef_size: size of the reef, NxM square grids, each  grid stores a solution
		# po: the rate between free/occupied at the beginning
		# Fb: BroadcastSpawner/ExistingCorals rate
		# Fa: fraction of corals duplicates its self and tries to settle in a different part of the reef
		# Fd: fraction of the worse health corals in reef will be applied depredation
		# Pd: Probabilty of depredation
		# k : number of attempts for a larvar to set in the reef.

		# reef: a maxtrix of dictionaries, each of those store a space's information (occupied/solution/health)
		# occupied_corals: position of occupied corals in reef-matrix (array 1dimension, each element store a position)
		# unselected_corals: corals in occupied_corals that aren't selected in broadcastSpawning
		# larvae: all larva ready to setting
		# sorted_health: a list of position, refer to position of each solution in reef-matrix, was sorted according to coral's health
        """
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = cro_paras["epoch"]
        self.pop_size = cro_paras["pop_size"]            # ~ number of space
        self.po = cro_paras["po"]
        self.Fb = cro_paras["Fb"]
        self.Fa = cro_paras["Fa"]
        self.Fd = cro_paras["Fd"]
        self.Pd_thres = cro_paras["Pd"]
        self.Pd = 0
        self.k = cro_paras["k"]
        self.G = cro_paras["G"]
        self.GCR = cro_paras["GCR"]

        self.G1 = cro_paras["G"][1]
        self.reef = np.array([])
        self.occupied_position = []  # after a gen, you should update the occupied_position
        self.alpha = 10 * self.Pd / self.epoch
        self.gama = 10 * (self.G[1] - self.G[0]) / self.epoch

    def _init_task_solution__(self, size):
        return np.random.uniform(self.domain_range[0], self.domain_range[1], size).tolist()

    # Init the coral reefs
    def _init_reef__(self):
        reef = np.array([{'occupied' : 0, 'solution' : [], 'health': self.HEALTH} for i in range(self.pop_size)])
        num_occupied = int(self.pop_size/(1 + self.po))
        occupied_position = random.sample(list(range(self.pop_size)), num_occupied)
        for i in (occupied_position):
            reef[i]['occupied'] = 1
            reef[i]['solution'] = self._init_task_solution__(self.problem_size)
            reef[i]['health'] = self._fitness_model__(reef[i]['solution'])
        self.occupied_position = occupied_position
        self.reef = reef

    # Update position which has been occupied by coral
    def _update_occupied_position__(self):
        self.occupied_position = []
        for i in range(self.pop_size):
            if self.reef[i]['occupied'] == 1:
                self.occupied_position.append(i)

    def _sort_occupied_position__(self):
        def referHealth(location):
            return self.reef[location]['health']
        self.occupied_position.sort(key = referHealth)

    def _broadcast_spawning_brooding__(self):
        # Step 1a
        self._update_occupied_position__()
        larvae = []
        num_of_occupied = len(self.occupied_position)
        selected_corals = random.sample(self.occupied_position, int(num_of_occupied*self.Fb))
        for i in self.occupied_position:
            if i not in selected_corals:
                p_solution = self.reef[i]['solution']
                blarva = self._gausion_mutation__(p_solution)
                larvae.append(blarva)
        # Step 1b
        while len(selected_corals) >= 2:
            p1, p2 = random.sample(selected_corals, 2)
            p1_solution = self.reef[p1]['solution']
            p2_solution = self.reef[p2]['solution']
            larva = self._multi_point_cross__(p1_solution, p2_solution)
            larvae.append(larva)
            selected_corals.remove(p1)
            selected_corals.remove(p2)
        self._larvae_setting__(larvae)

    def _larvae_setting__(self, larvae):
        for larva in larvae:
            larva_fit = self._fitness_model__(larva)
            for i in range(self.k):
                p = random.randint(0, self.pop_size - 1)
                if self.reef[p]['occupied'] == 0 or self.reef[p]['health'] > larva_fit:
                    self.reef[p]['occupied'] = 1
                    self.reef[p]['solution'] = larva
                    self.reef[p]['health'] = larva_fit
                    break

    def _budding_setting__(self, larvae):
        for larva in larvae:
            for i in range(self.k):
                p = random.randint(0, self.pop_size - 1)
                if self.reef[p]['occupied'] == 0 or (self.reef[p]['health'] > larva['health']) :
                    self.reef[p] = larva.copy()
                    break

    def _asexual_reproduction__(self):
        self._update_occupied_position__() # Cap nhat lai vi tri
        self._sort_occupied_position__()

        num_duplicate = int(len(self.occupied_position)*self.Fa)
        selected_duplicator = self.occupied_position[:num_duplicate]
        duplicated_corals = np.take(self.reef, selected_duplicator)
        self._budding_setting__(duplicated_corals)

    def _depredation__(self):
        rate = random.random()
        if rate < self.Pd:
            num__depredation__ = int(len(self.occupied_position) * self.Fd)
            selected_depredator = self.occupied_position[-num__depredation__:]
            for pos in selected_depredator:
                self.reef[pos]['occupied'] = 0

    ### Crossover
    def _multi_point_cross__(self, parent1, parent2):
        p1, p2 = random.sample(list(range(len(parent1))), 2)
        start = min(p1, p2)
        end = max(p1, p2)
        return parent1[:start] + parent2[start:end] + parent1[end:]

    ### Mutation
    def _swap_mutation__(self, solution):
        new_sol = solution.copy()
        new_sol[random.randint(0, self.problem_size-1)] = np.random.uniform(self.domain_range[0], self.domain_range[1])
        return new_sol

    def _gausion_mutation__(self, solution):
        new_sol = np.array(solution)
        rd = np.random.uniform(0, 1, self.problem_size)
        new_sol[rd < self.GCR] = (new_sol + self.G1*(self.domain_range[1] - self.domain_range[0])* np.random.normal(np.zeros(self.problem_size), 1))[rd < self.GCR]
        new_sol[new_sol < self.domain_range[0]] = self.domain_range[0]
        new_sol[new_sol > self.domain_range[1]] = self.domain_range[1]
        return new_sol.tolist()

    def _train__(self):
        best_train = {"occupied": 0, "solution": None, "health": self.HEALTH}
        self._init_reef__()
        for j in range(0, self.epoch):
            self._broadcast_spawning_brooding__()
            self._asexual_reproduction__()
            self._depredation__()
            if self.Pd <= self.Pd_thres:
                self.Pd += self.alpha
            if self.G1 >= self.G[0]:
                self.G1 -= self.gama
            bes_pos = self.occupied_position[0]
            bes_sol = self.reef[bes_pos]
            if bes_sol['health'] < best_train["health"]:
                best_train = bes_sol
            if self.print_train:
                print("> Epoch {}: Best training fitness {}".format(j + 1, best_train["health"]))
            self.loss_train.append(best_train["health"])

        return best_train["solution"], self.loss_train, best_train['health']


class OCRO(BaseCRO):
    """
    This is a variant of CRO which is combined Opposition-based learning and Coral Reefs Optimization
    """

    def __init__(self, root_algo_paras=None, cro_paras = None, ocro_paras=None):
        BaseCRO.__init__(self, root_algo_paras, cro_paras)
        self.best_sol = []
        self.restart_count = ocro_paras["restart_count"]

    def __local_seach__(self):
        reef = np.array([{'occupied' : 0, 'solution' : [], 'health': self.HEALTH} for i in range(self.pop_size)])
        num_occupied = int(self.pop_size/(1 + self.po))
        occupied_position = random.sample(list(range(self.pop_size)), num_occupied)
        for i in (occupied_position):
            reef[i]['occupied'] = 1
            reef[i]['solution'] = np.array(self._init_task_solution__(self.problem_size))
            rd = np.random.uniform(0, 1, self.problem_size)
            reef[i]['solution'][rd < 0.1] = self.best_sol[rd < 0.1]
            reef[i]['solution'] = reef[i]['solution'].tolist()
            reef[i]['health'] = self._fitness_model__(reef[i]['solution'])
        self.occupied_position = occupied_position
        self.reef = reef

    def __opposition_based_larva__(self, budding_pos):
        larvae = []
        for i in budding_pos:
            Olarva = self.reef[i].copy()
            Olarva['solution'] = (self.domain_range[1] + self.domain_range[0] - self.best_sol + random.random()*(self.best_sol - np.array(self.reef[i]['solution']))).tolist()
            Olarva['health'] = self._fitness_model__(Olarva['solution'])
            if Olarva['health'] < self.reef[i]['health']:
                larvae.append(Olarva)
        return larvae

    def _asexual_reproduction__(self):
        self._update_occupied_position__()
        self._sort_occupied_position__()
        self.best_coral = self.reef[self.occupied_position[0]]
        num_duplicate = int(len(self.occupied_position) * self.Fa)
        selected_duplicator = self.occupied_position[:num_duplicate]
        duplicated_corals = np.take(self.reef, selected_duplicator)
        # duplicated_corals = self.__opposition_based_larva__(selected_duplicator)
        self._budding_setting__(duplicated_corals)
        # self.update_occupied_position()
        # self.sort_occupied_position()

    def _depredation__(self):
        # rate = random.random()
        # if rate < self.Pd:
        num_depredation = int(len(self.occupied_position) * self.Fd)
        selected_depredator = random.sample(self.occupied_position[-self.pop_size//2:], num_depredation)
        for i in selected_depredator:
            Olarva = self.reef[i].copy()
            Olarva['solution'] = (self.domain_range[1] + self.domain_range[0] - self.best_sol + np.random.uniform(0,1, self.problem_size)*(self.best_sol - np.array(self.reef[i]['solution']))).tolist()
            Olarva['health'] = self._fitness_model__(Olarva['solution'])
            if Olarva['health'] < self.reef[i]['health']:
                self.reef[i] = Olarva.copy()
            else:
                self.reef[i]['occupied'] = 0

    def _train__(self):
        best_train = {"occupied": 0, "solution": None, "health": self.HEALTH}
        self._init_reef__()
        self.best_sol = np.array(self.reef[self.occupied_position[0]]['solution']).copy()
        reset_count = 0
        for j in range(0, self.epoch):
            self._broadcast_spawning_brooding__()
            self._asexual_reproduction__()
            self._depredation__()
            if self.Pd <= self.Pd_thres:
                self.Pd += self.alpha
            if self.G1 >= self.G[0]:
                self.G1 -= self.gama
            bes_sol = self.reef[self.occupied_position[0]]
            reset_count += 1
            if reset_count == self.restart_count:
                self._init_reef__()
                self.reef[0] = self.best_coral
                reset_count = 0
            if bes_sol['health'] < best_train["health"]:
                best_train = bes_sol
                reset_count = 0
            if self.print_train:
                print("> Epoch {}: Best training fitness {}".format(j + 1, best_train["health"]))
            self.loss_train.append(best_train["health"])

        return best_train["solution"], self.loss_train
