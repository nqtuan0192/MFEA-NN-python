from MFEATask import *
from datetime import datetime

class Chromosome:
    global TRAINING_SIZE
    global TESTING_SIZE

    global NUMBEROF_TASKS
    global NUMBEROF_LAYERS
    global TASKS
    global TASKS_MAX

    global NUMBEROF_INPUT
    global NUMBEROF_OUTPUT


    scalar_fitness = -1.0   # float
    skill_factor = 0        # uint

    factorial_costs = []    # list of float, size = numberof_tasks
    factorial_rank = []     # list of uint, size = numberof_tasks
    accuracy = []           # list of float, size = numberof_tasks

    parameters = {}                  # weights and biases, dictionary of np.ndarray, size = numberof_layers

    def __init__(self):
        print('init')
        '''L = len(TASK_MAX)
        for layer in range(1, L):
            self.parameters['W' + str(layer)] = np.ndarray((TASK_MAX[layer], TASK_MAX[layer - 1]))
            print(self.parameters['W' + str(layer)].shape)
            self.parameters['b' + str(layer)] = np.ndarray((TASK_MAX[layer], 1))
            print(self.parameters['b' + str(layer)].shape)'''

    def initialize_parameters(self):
        #np.random.seed(datetime.now())
        L = len(TASK_MAX)
        for layer in range(1, L):
            self.parameters['W' + str(layer)] = np.random.rand(TASK_MAX[layer], TASK_MAX[layer - 1])
            self.parameters['b' + str(layer)] = np.random.rand(TASK_MAX[layer], 1)

    def print_parameters(self):
        L = len(TASK_MAX)
        for layer in range(1, L):
            print('W' + str(layer), self.parameters['W' + str(layer)].shape, self.parameters['W' + str(layer)])
            print('b' + str(layer), self.parameters['b' + str(layer)].shape, self.parameters['b' + str(layer)])

    def forward_eval(self):
        L = len(TASK_MAX)
#end class Chromosome

def sbx_beta_transform(array, distribution_index):
    ret = np.ndarray(array.shape)
    ret[] =
	for i in range(len(array)):
		if array[i] <= 0.5:
			ret.append((2.0 * array[i]) ** (1.0 / (distribution_index + 1)))
		else:
			ret.append((1.0 / (2.0 * (1.0 - array[i]))) ** (1.0 / (distribution_index + 1.0)))
	return ret

def sbx_children_generate(p1, p2, v_rand):
	c1 = []
	c2 = []
	for i in range(len(p1)):
		c1.append(0.5 * ((1 + v_rand[i]) * p1[i] + (1 - v_rand[i]) * p2[i]))
		c2.append(0.5 * ((1 - v_rand[i]) * p1[i] + (1 + v_rand[i]) * p2[i]))
	return (c1, c2)

def pmu_children_generate(p, r, mu_ratio, pmu_index):
	c = []
	for i in range(len(p)):
		if r[i] <= mu_ratio:
			if r[i] / mu_ratio <= 0.5:
				ddd = (2 * r[i]) ** (1.0 / (2.0 + pmu_index)) - 1
				c.append(p[i] + ddd * p[i])
			else:
				ddd = 1.0 - (2.0 * (1.0 - r[i])) ** (1.0 / (1.0 + pmu_index))
				c.append(p[i] + ddd * (1.0 - p[i]))
		else:
			c.append(p[i])
	return c