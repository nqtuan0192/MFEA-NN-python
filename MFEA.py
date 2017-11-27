import random
import numpy as np

from Chromosome import *

import MFEATask as mfeatask



class MFEA:
    def __init__(self, pop_size = 90, gen_size = 1000):
        self.population_size = pop_size
        self.generation_size = gen_size
        self.population = []

        for i in range(pop_size):
            idv = Chromosome()
            idv.initialize_parameters()
            idv.skill_factor = i % mfeatask.NUMBEROF_TASKS
            self.population.append(idv)


    def evolution(self):
        # tunable factors
        cf_distributionindex = 2.0              # crossover factor, index of Simulated Binary Crossover
        mf_randommatingprobability = 1.0        # mutation factor, random mating probability
        mf_polynomialmutationindex = 5.0        # mutation factor, index of Polynomial Mutation Operator
        mf_mutationratio = 0.5                 # mutation factor,

        random.shuffle(self.population)
        for i in range(int(self.population_size / 2)):
            idv1 = self.population[i]
            idv2 = self.population[i + int(self.population_size / 2)]
            if idv1.skill_factor == idv2.skill_factor or np.random.rand() < mf_randommatingprobability:
                print('test', i, i + int(self.population_size / 2))
                child1, child2 = Chromosome(), Chromosome()

                op_crossover(idv1, idv2, cf_distributionindex, child1, child2)

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

                op_uniformcrossover(child1, child2)
            else:
                print('never get here')



