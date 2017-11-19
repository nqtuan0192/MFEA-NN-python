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

    #parameters = {}                  # weights and biases, dictionary of np.ndarray, size = numberof_layers

    def __init__(self):
        print('init')

        self.scalar_fitness = -1.0  # float
        self.skill_factor = 0       # uint

        self.factorial_costs = []   # list of float, size = numberof_tasks
        self.factorial_rank = []    # list of uint, size = numberof_tasks
        self.accuracy = []          # list of float, size = numberof_tasks

        self.parameters = {}        # weights and biases, dictionary of np.ndarray, size = numberof_layers

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
    ret = np.empty(array.shape)
    idx = array <= 0.5
    ret[idx] = (2.0 * array[idx]) ** (1.0 / (distribution_index + 1))
    idx = array > 0.5 # need check
    ret[idx] = (1.0 / (2.0 * (1.0 - array[idx]))) ** (1.0 / (distribution_index + 1.0))
    return ret

def sbx_children_generate(p1, p2, v_rand):
	c1 = 0.5 * ((1 + v_rand) * p1 + (1 - v_rand) * p2)
	c2 = 0.5 * ((1 - v_rand) * p1 + (1 + v_rand) * p2)
	return (c1, c2)

def test_sbx_operator(loop):
    np.random.seed(9)
    p1 = np.random.rand(1, 10)
    p2 = np.random.rand(1, 10)
    (c1, c2) = (p1, p2)
    for i in range(loop):
        r = np.random.rand(1, 10)
        rsbx = sbx_beta_transform(r, 3)
        (c1, c2) = sbx_children_generate(c1, c2, rsbx)
    print('diff = ', p1 + p2 == c1 + c2)
    print('p1 + p2 = ', p1 + p2)
    print('c1 + c2 = ', c1 + c2)
    print('c1 = ', c1)
    print('c2 = ', c2)

    return (c1, c2)


def pmu_children_generate(p, r, mu_ratio, pmu_index):
    c = np.empty(p.shape)

    rp = p.ravel()
    rc = c.ravel()
    rr = r.ravel()

    for i in range(len(rp)):
        if (rr[i] <= mu_ratio):
            u = np.random.rand()
            if (u <= 0.5):
                ddd = (2 * u) ** (1.0 / (2.0 + pmu_index)) - 1
                rc[i] = rp[i] + ddd * rp[i]
            else:
                ddd = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (1.0 + pmu_index))
                rc[i] = rp[i] + ddd * (1.0 - rp[i])
        else:
            rc[i] = rp[i]
    return c

def test_pmu_operator(loop):
    np.random.seed(1)
    p = np.random.rand(1, 10)
    c = p
    for i in range(loop):
        r = np.random.rand(1, 10)
        c = pmu_children_generate(c, r, 0.05, 3)
    print('p = ', p)
    print('c = ', c)
    print('diff = ', p == c)

    return c



def op_crossover(chromo1, chromo2, cf_distributionindex):
    (child1, child2) = Chromosome(), Chromosome()
    L = len(TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        r = np.random.random(chromo1.parameters['W' + str(layer)].shape)
        r = sbx_beta_transform(r, cf_distributionindex)
        (child1.parameters['W' + str(layer)], child2.parameters['W' + str(layer)]) = sbx_children_generate(
            chromo1.parameters['W' + str(layer)], chromo2.parameters['W' + str(layer)], r)

        r = np.random.random(chromo1.parameters['b' + str(layer)].shape)
        r = sbx_beta_transform(r, cf_distributionindex)
        (child1.parameters['b' + str(layer)], child2.parameters['b' + str(layer)]) = sbx_children_generate(
            chromo1.parameters['b' + str(layer)], chromo2.parameters['b' + str(layer)], r)
    return (child1, child2)

def op_mutate(chromo, mf_polynomialmutationindex, mf_mutationratio):
    child = Chromosome()
    L = len(TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        r = np.random.random(chromo.parameters['W' + str(layer)].shape)
        child.parameters['W' + str(layer)] = pmu_children_generate(chromo.parameters['W' + str(layer)], r,
                                                                   mf_mutationratio, mf_polynomialmutationindex)
        r = np.random.random(chromo.parameters['b' + str(layer)].shape)
        child.parameters['b' + str(layer)] = pmu_children_generate(chromo.parameters['b' + str(layer)], r,
                                                                   mf_mutationratio, mf_polynomialmutationindex)
    return child

def test_op_crossover():
    idv1 = Chromosome()
    idv1.initialize_parameters()

    idv2 = Chromosome()
    idv2.initialize_parameters()

    (child1, child2) = op_crossover(idv1, idv2, 3.0)

    L = len(TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        print('weight check', layer, (idv1.parameters['W' + str(layer)] + idv2.parameters['W' + str(layer)]) - (
        child1.parameters['W' + str(layer)] + child2.parameters['W' + str(layer)]))
        print('biases check', layer, (idv1.parameters['b' + str(layer)] + idv2.parameters['b' + str(layer)]) - (
        child1.parameters['b' + str(layer)] + child2.parameters['b' + str(layer)]))

def test_op_mutate():
    idv = Chromosome()
    idv.initialize_parameters()

    child = op_mutate(idv, 1.0, 0.5)

    idv.print_parameters()
    child.print_parameters()

    L = len(TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        print('weight diff', layer, idv.parameters['W' + str(layer)] - child.parameters['W' + str(layer)])
        print('biases diff', layer, idv.parameters['b' + str(layer)] - child.parameters['b' + str(layer)])