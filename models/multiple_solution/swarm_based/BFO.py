import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from models.multiple_solution.root_multiple import RootAlgo

class BaseBFO(RootAlgo):
    """
    Basic version of Bacterial Foraging Optimization Algorithm: Taken from here
    http://www.cleveralgorithms.com/nature-inspired/swarm/bfoa.html
    """
    ID_VECTOR = 0
    ID_COST = 1
    ID_INTER = 2
    ID_FITNESS = 3
    ID_SUM_NUTRIENTS = 4

    def __init__(self, root_algo_paras=None, bfo_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        # algorithm configuration
        self.pop_size = bfo_paras["pop_size"]
        self.step_size = bfo_paras["Ci"]         # Ci
        self.p_eliminate = bfo_paras["Ped"]     # Ped
        self.swim_length = bfo_paras["Ns"]     # Ns

        self.elim_disp_steps = bfo_paras["Ned"]  # Ned
        self.repro_steps = bfo_paras["Nre"]     # Nre
        self.chem_steps = bfo_paras["Nc"]       # Nc

        self.d_attr = bfo_paras["attract_repel"][0]
        self.w_attr = bfo_paras["attract_repel"][1]
        self.h_rep = bfo_paras["attract_repel"][2]
        self.w_rep = bfo_paras["attract_repel"][3]

    def _create_solution__(self, minmax=None):
        vector = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        cost = 0.0
        interaction = 0.0
        fitness = 0.0
        sum_nutrients = 0.0
        return [vector, cost, interaction, fitness, sum_nutrients]

    def _compute_cell_interaction__(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem_size * mean_squared_error(cell[self.ID_VECTOR], other[self.ID_VECTOR])
            sum_inter += d * np.exp(w * diff)
        return sum_inter

    def _attract_repel__(self, cell, cells):
        attract = self._compute_cell_interaction__(cell, cells, -self.d_attr, -self.w_attr)
        repel = self._compute_cell_interaction__(cell, cells, self.h_rep, -self.w_rep)
        return attract + repel

    def _evaluate__(self, cell, cells):
        cell[self.ID_COST] = self._fitness_model__(cell[self.ID_VECTOR])
        cell[self.ID_INTER] = self._attract_repel__(cell, cells)
        cell[self.ID_FITNESS] = cell[self.ID_COST] + cell[self.ID_INTER]

    def _tumble_cell__(self, cell, step_size):
        delta_i = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        unit_vector = delta_i / np.sqrt(np.dot(delta_i, delta_i.T))
        vector = cell[self.ID_VECTOR] + step_size * unit_vector
        return [vector, 0.0, 0.0, 0.0, 0.0]

    def _chemotaxis__(self, l=None, k = None, cells=None):
        current_best = None
        for j in range(0, self.chem_steps):
            moved_cells = []                            # New generation
            for i, cell in enumerate(cells):
                sum_nutrients = 0.0
                self._evaluate__(cell, cells)
                if (current_best is None) or cell[self.ID_COST] < current_best[self.ID_COST]:
                    current_best = deepcopy(cell)
                sum_nutrients += cell[self.ID_FITNESS]

                for m in range(0, self.swim_length):
                    new_cell = self._tumble_cell__(cell, self.step_size)
                    self._evaluate__(new_cell, cells)
                    if current_best[self.ID_COST] > new_cell[self.ID_COST]:
                        current_best = deepcopy(new_cell)
                    if new_cell[self.ID_FITNESS] > cell[self.ID_FITNESS]:
                        break
                    cell = deepcopy(new_cell)
                    sum_nutrients += cell[self.ID_FITNESS]

                cell[self.ID_SUM_NUTRIENTS] = sum_nutrients
                moved_cells.append(deepcopy(cell))
            cells = deepcopy(moved_cells)
            self.loss_train.append(current_best[self.ID_COST])
            if self.print_train:
                print("elim = %d, repro = %d, chemo = %d >> best fitness=%.8f" %(l + 1, k + 1, j + 1, current_best[self.ID_COST]))
        return current_best, cells

    def _train__(self):
        cells = [self._create_solution__(minmax=0) for _ in range(0, self.pop_size)]
        half_pop_size = int(self.pop_size / 2)
        best = None
        for l in range(0, self.elim_disp_steps):
            for k in range(0, self.repro_steps):
                current_best, cells = self._chemotaxis__(l, k, cells)
                if (best is None) or current_best[self.ID_COST] < best[self.ID_COST]:
                    best = current_best
                cells = sorted(cells, key=lambda cell: cell[self.ID_SUM_NUTRIENTS])
                cells = deepcopy(cells[0:half_pop_size]) + deepcopy(cells[0:half_pop_size])
            for idc in range(self.pop_size):
                if np.random.uniform() < self.p_eliminate:
                    cells[idc] = self._create_solution__(minmax=0)
        return best[self.ID_VECTOR], self.loss_train



class ABFOLS(RootAlgo):
    """
    ### This is the best improvement version of BFO
    ## Paper: An Adaptive Bacterial Foraging Optimization Algorithm with Lifecycle and Social Learning
    """
    ID_VECTOR = 0
    ID_FITNESS = 1
    ID_NUTRIENT = 2
    ID_PERSONAL_BEST = 3

    NUMBER_CONTROL_RATE = 2

    def __init__(self, root_algo_paras=None, abfols_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        # algorithm configuration
        self.epoch = abfols_paras["epoch"]
        self.pop_size = abfols_paras["pop_size"]
        self.step_size = abfols_paras["Ci"]                     # Ci
        self.p_eliminate = abfols_paras["Ped"]                 # Ped
        self.swim_length = abfols_paras["Ns"]                 # Ns

        self.N_adapt = abfols_paras["N_minmax"][0]  # Dead threshold value
        self.N_split = abfols_paras["N_minmax"][1]                          # split threshold value
        self.C_s = self.step_size[0] * (self.domain_range[1] - self.domain_range[0])
        self.C_e = self.step_size[1] * (self.domain_range[1] - self.domain_range[0])

    def _create_solution__(self, minmax=None):
        vector = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(vector, self.ID_MAX_PROBLEM)          # Current position
        nutrient = 0          # total nutrient gained by the bacterium in its whole searching process.(int number)
        p_best = deepcopy(vector)
        return [vector, fitness, nutrient, p_best]

    def _tumble_cell__(self, cell=None, step_size=None):
        delta_i = (self.global_best[self.ID_VECTOR] - cell[self.ID_VECTOR]) + (cell[self.ID_PERSONAL_BEST] - cell[self.ID_VECTOR])
        if np.all(delta_i == 0):
            delta_i = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        unit_vector = delta_i / np.sqrt(np.dot(delta_i, delta_i.T))

        #unit_vector = np.random.uniform() * 1.2 * (self.global_best[self.ID_VECTOR] - cell[self.ID_VECTOR]) + np.random.uniform() *1.2* (cell[self.ID_PERSONAL_BEST] - cell[self.ID_VECTOR])
        vec = cell[self.ID_VECTOR] + step_size * unit_vector
        fit = self._fitness_model__(vec, self.ID_MAX_PROBLEM)
        if fit> cell[self.ID_FITNESS]:
            cell[self.ID_NUTRIENT] += 1
        else:
            cell[self.ID_NUTRIENT] -= 1

        if fit > cell[self.ID_FITNESS]:             # Update personal best
            cell[self.ID_PERSONAL_BEST] = deepcopy(vec)
        cell[self.ID_VECTOR] = deepcopy(vec)
        cell[self.ID_FITNESS] = fit

        # Update global best
        self.global_best = deepcopy(cell) if self.global_best[self.ID_FITNESS] < cell[self.ID_FITNESS] else self.global_best
        return cell


    def _update_step_size__(self, cells=None, id=None):
        total_fitness = sum(temp[self.ID_FITNESS] for temp in cells)
        step_size = self.C_s - (self.C_s - self.C_e) * cells[id][self.ID_FITNESS]/ total_fitness
        step_size = step_size / cells[id][self.ID_NUTRIENT] if cells[id][self.ID_NUTRIENT] > 0 else step_size
        return step_size

    def _train__(self):
        cells = [self._create_solution__(minmax=0) for _ in range(0, self.pop_size)]
        self.global_best = self._get_global_best__(cells, self.ID_FITNESS, self.ID_MAX_PROBLEM)

        for loop in range(self.epoch):
            i = 0
            while i < len(cells):
                step_size = self._update_step_size__(cells, i)
                JLast = cells[i][self.ID_FITNESS]
                cells[i] = self._tumble_cell__(cell=cells[i], step_size=step_size)
                m = 0
                while m < self.swim_length:             # Ns
                    if cells[i][self.ID_FITNESS] < JLast:
                        step_size = self._update_step_size__(cells, i)
                        JLast = cells[i][self.ID_FITNESS]
                        cells[i] = self._tumble_cell__(cell=cells[i], step_size=step_size)
                        m += 1
                    else:
                        m = self.swim_length

                S_current = len(cells)
                #print("======= Current Nutrient: {}".format(cells[i][self.ID_NUTRIENT]))

                if cells[i][self.ID_NUTRIENT] > max(self.N_split, self.N_split + (S_current - self.pop_size) / self.N_adapt):
                    new_cell = deepcopy(cells[i])
                    new_cell[self.ID_NUTRIENT] = 0
                    cells[i][self.ID_NUTRIENT] = 0
                    cells.append(new_cell)
                    break

                if cells[i][self.ID_NUTRIENT] < min(self.NUMBER_CONTROL_RATE, self.NUMBER_CONTROL_RATE + (S_current - self.pop_size) / self.N_adapt):
                    cells.pop(i)
                    i -= 1
                    break

                if cells[i][self.ID_NUTRIENT] < self.NUMBER_CONTROL_RATE and np.random.uniform() < self.p_eliminate:
                    temp = self._create_solution__(minmax=0)
                    self.global_best = deepcopy(temp) if temp[self.ID_FITNESS] > self.global_best[self.ID_FITNESS] else self.global_best
                    cells[i] = temp
                i += 1
            self.loss_train.append([1.0 / self.global_best[self.ID_FITNESS], 1.0 / self.global_best[self.ID_FITNESS]])
            if self.print_train:
                print("Epoch = {}, Pop_size = {}, >> Best fitness = {}".format(loop + 1, len(cells), 1.0 / self.global_best[self.ID_FITNESS]))

        return self.global_best[self.ID_VECTOR], self.loss_train, self.global_best[self.ID_FITNESS]
