import random
import numpy as np
import matplotlib.pyplot as plt

from Chromosome import *

import MFEATask as mfeatask



class MFEA:
    def __init__(self, X_train, Y_train, X_test, Y_test, pop_size = 90, gen_size = 1000):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.population_size = pop_size
        self.generation_size = gen_size
        self.population = 2 * pop_size * [None]

        self.mse = {}
        for task in range(mfeatask.NUMBEROF_TASKS):
            self.mse['mse' + str(task)] = []

        for i in range(pop_size):
            self.population[i], self.population[i + pop_size] = Chromosome(), Chromosome()
            self.population[i].initialize_parameters()
            self.population[i].skill_factor = i % mfeatask.NUMBEROF_TASKS
            self.population[i].forward_eval(self.X_train, self.Y_train)


    def evolution(self):
        # tunable factors
        cf_distributionindex = 2.0              # crossover factor, index of Simulated Binary Crossover
        mf_randommatingprobability = 1.0        # mutation factor, random mating probability
        mf_polynomialmutationindex = 5.0        # mutation factor, index of Polynomial Mutation Operator
        mf_mutationratio = 0.5                  # mutation factor,

        generation = 0
        while generation < self.generation_size:
            generation += 1
            print('generation', generation)

            random.shuffle(self.population[:self.population_size])
            for i in range(int(self.population_size / 2)):
                idv1 = self.population[i]
                idv2 = self.population[i + int(self.population_size / 2)]
                child1 = self.population[i + self.population_size]
                child2 = self.population[i + self.population_size + int(self.population_size / 2)]
                #print('test', i, i + int(self.population_size / 2), i + self.population_size, i + self.population_size + int(self.population_size / 2))

                if idv1.skill_factor == idv2.skill_factor or np.random.rand() < mf_randommatingprobability:
                    # print('test', i, i + int(self.population_size / 2))

                    # simulated binary crossover
                    op_crossover(idv1, idv2, cf_distributionindex, child1, child2)

                    # polynomial mutation operator
                    op_mutate(child1, mf_polynomialmutationindex, mf_mutationratio, child1)
                    op_mutate(child2, mf_polynomialmutationindex, mf_mutationratio, child2)

                    # probabilistic assign the skill factor for children from their parents
                    if (np.random.rand() <= 0.5):
                        child1.skill_factor = idv1.skill_factor
                    else:
                        child1.skill_factor = idv2.skill_factor
                    if (np.random.rand() <= 0.5):
                        child2.skill_factor = idv1.skill_factor
                    else:
                        child2.skill_factor = idv2.skill_factor

                    # Uniform crossover-like variable swap between two new born children
                    op_uniformcrossover(child1, child2)
                else:
                    print('never get here')

                child1.forward_eval(self.X_train, self.Y_train)
                child2.forward_eval(self.X_train, self.Y_train)

            # update rank for each task
            for task in range(mfeatask.NUMBEROF_TASKS):
                self.population.sort(key=lambda chromo: chromo.factorial_costs[task])
                for i, idv in enumerate(self.population):
                    idv.factorial_rank[task] = i + 1

                self.mse['mse' + str(task)].append(self.population[0].factorial_costs[task])

            # calculate scala fitness
            for idv in self.population:
                idv.scalar_fitness = 1.0 / np.min(idv.factorial_rank)

            self.population.sort(key=lambda chromo: chromo.scalar_fitness, reverse=True)


    def sumarizeTrainingStep(self):
        x = np.arange(0, self.generation_size, 1)
        color = ['blue', 'green', 'red', 'yellow', 'black']
        for task in range(mfeatask.NUMBEROF_TASKS):
            plt.plot(x, self.mse['mse' + str(task)], color[task])
        plt.show()


    def revalAccuracyOnTestingData(self):
        print('On training set:')
        for idv in self.population:
            idv.forward_eval(self.X_train, self.Y_train, is_eval_acc=True)
        for task in range(mfeatask.NUMBEROF_TASKS):
            self.population.sort(key=lambda chromo: chromo.factorial_costs[task])
            print('--- Task ', task, ' best mse = ', self.population[0].factorial_costs[task])
            print('         ', task, ' best acc = ', self.population[0].accuracy[task])
        print('On testing set:')
        for idv in self.population:
            idv.forward_eval(self.X_test, self.Y_test, is_eval_acc=True)
        for task in range(mfeatask.NUMBEROF_TASKS):
            self.population.sort(key=lambda chromo: chromo.factorial_costs[task])
            print('--- Task ', task, ' best mse = ', self.population[0].factorial_costs[task])
            print('         ', task, ' best acc = ', self.population[0].accuracy[task])