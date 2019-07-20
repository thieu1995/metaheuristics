import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo

class BasePSO(RootAlgo):
    ID_CURRENT_POSITION = 0
    ID_PAST_POSITION_BEST = 1
    ID_VECTOR_V = 2
    ID_CURRENT_FITNESS = 3
    ID_PAST_FITNESS = 4

    def __init__(self, root_algo_paras=None, pso_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = pso_paras["epoch"]
        self.pop_size = pso_paras["pop_size"]
        self.c1 = pso_paras["c_minmax"][0]
        self.c2 = pso_paras["c_minmax"][1]
        self.w_min = pso_paras["w_minmax"][0]
        self.w_max = pso_paras["w_minmax"][1]

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

    def _train__(self):
        """
        - Khoi tao quan the (tinh ca global best)
        - Di chuyen va update vi tri, update gbest
        """
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        gbest = self._get_global_best__(pop=pop, id_fitness=self.ID_CURRENT_FITNESS, id_best=self.ID_MIN_PROBLEM)

        for i in range(self.epoch):
            # Update weight after each move count  (weight down)
            w = (self.epoch - i) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for j in range(self.pop_size):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                #%% testing
                vi_sau = w * pop[j][self.ID_VECTOR_V] + self.c1 * r1 * \
                         (pop[j][self.ID_PAST_POSITION_BEST] - pop[j][self.ID_CURRENT_POSITION]) \
                         + self.c2 * r2 * (gbest[self.ID_CURRENT_POSITION] - pop[j][self.ID_CURRENT_POSITION])

                #%% testing 2
                xi_sau = pop[j][self.ID_CURRENT_POSITION] + vi_sau                 # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                fit_sau = self._fitness_model__(solution=xi_sau, minmax=0)
                fit_truoc = pop[j][self.ID_PAST_FITNESS]

                # Cap nhat x hien tai, v hien tai, so sanh va cap nhat x past best voi x hien tai
                pop[j][self.ID_CURRENT_POSITION] = deepcopy(xi_sau)
                pop[j][self.ID_VECTOR_V] = deepcopy(vi_sau)
                pop[j][self.ID_CURRENT_FITNESS] = fit_sau

                if fit_sau < fit_truoc:
                    pop[j][self.ID_PAST_POSITION_BEST] = deepcopy(xi_sau)
                    pop[j][self.ID_PAST_FITNESS] = fit_sau

            current_best = self._get_global_best__(pop=pop, id_fitness=self.ID_CURRENT_FITNESS, id_best=self.ID_MIN_PROBLEM)
            if current_best[self.ID_CURRENT_FITNESS] < gbest[self.ID_CURRENT_FITNESS]:
                gbest = deepcopy(current_best)
            self.loss_train.append(gbest[self.ID_CURRENT_FITNESS])
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i+1, gbest[self.ID_CURRENT_FITNESS]))

        return gbest[self.ID_CURRENT_POSITION], self.loss_train, gbest[self.ID_CURRENT_FITNESS]



