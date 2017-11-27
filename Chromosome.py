import MFEATask as mfeatask
from datetime import datetime
import matplotlib.pyplot as plt

from InputHandler import *

def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_tanh(x):
    return np.tanh(x)


def np_relu(x):
    return np.maximum(0, x)

class Chromosome:

    def __init__(self):
        self.scalar_fitness = -1.0  # float
        self.skill_factor = 0  # uint

        self.factorial_costs = mfeatask.NUMBEROF_TASKS * [None]  # list of float, size = numberof_tasks
        self.factorial_rank = mfeatask.NUMBEROF_TASKS * [None]  # list of uint, size = numberof_tasks
        self.accuracy = mfeatask.NUMBEROF_TASKS * [None]  # list of float, size = numberof_tasks

        self.parameters = {}  # weights and biases, dictionary of np.ndarray, size = numberof_layers

    def initialize_parameters(self):
        # np.random.seed(datetime.now())
        L = len(mfeatask.TASK_MAX)
        for layer in range(1, L):
            self.parameters['W' + str(layer)] = np.random.rand(mfeatask.TASK_MAX[layer], mfeatask.TASK_MAX[layer - 1])
            self.parameters['b' + str(layer)] = np.random.rand(mfeatask.TASK_MAX[layer], 1)

    def print_parameters(self):
        L = len(mfeatask.TASK_MAX)
        for layer in range(1, L):
            print('W' + str(layer), self.parameters['W' + str(layer)].shape, self.parameters['W' + str(layer)])
            print('b' + str(layer), self.parameters['b' + str(layer)].shape, self.parameters['b' + str(layer)])

    def forward_eval(self, X, Y):
        L = mfeatask.TASKS_LAYERSIZE[self.skill_factor]
        layers = mfeatask.TASKS[self.skill_factor]

        A = X
        for l in range(1, L):
            Wl = self.parameters['W' + str(l)].flatten()
            Wl = Wl[0:layers[l] * layers[l - 1]].reshape(layers[l], layers[l - 1])
            bl = self.parameters['b' + str(l)].flatten()
            bl = bl[0:layers[l]].reshape(layers[l], 1)

            Z = np.dot(Wl, A) + bl
            A = np_relu(Z)

        WL = self.parameters['W' + str(L)].flatten()
        WL = WL[0:layers[L] * layers[L - 1]].reshape(layers[L], layers[L - 1])
        bL = self.parameters['b' + str(L)].flatten()
        bL = bL[0:layers[L]].reshape(layers[L], 1)

        ZL =  np.dot(WL, A) + bL
        AL = np_sigmoid(ZL)

        cost = 0.5 * np.mean((Y - AL) ** 2) # 1 / (2 * m) * (Y - Ypredict)^2

        lambd = 0.1
        m = Y.shape[1]
        L2_regularization_cost = 0
        for l in range(1, L + 1):
            L2_regularization_cost = L2_regularization_cost + np.sum(np.square(self.parameters['W' + str(l)]))
        L2_regularization_cost = lambd / (2 * m) * L2_regularization_cost

        self.factorial_costs[self.skill_factor] = cost
        return cost # + L2_regularization_cost



# end class Chromosome

def sbx_beta_transform(array, distribution_index):
    ret = np.empty(array.shape)
    idx = array <= 0.5
    ret[idx] = (2.0 * array[idx]) ** (1.0 / (distribution_index + 1))
    idx = array > 0.5  # need check
    ret[idx] = (1.0 / (2.0 * (1.0 - array[idx]))) ** (1.0 / (distribution_index + 1.0))
    return ret


def sbx_children_generate(p1, p2, v_rand):
    c1 = 0.5 * ((1 + v_rand) * p1 + (1 - v_rand) * p2)
    c2 = 0.5 * ((1 - v_rand) * p1 + (1 + v_rand) * p2)
    return (c1, c2)


def test_sbx_operator(loop):
    np.random.seed(1)
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

    return c1, c2


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


def op_crossover(chromo1, chromo2, cf_distributionindex, child1, child2):
    L = len(mfeatask.TASK_MAX)
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


def op_mutate(chromo, mf_polynomialmutationindex, mf_mutationratio, child):
    L = len(mfeatask.TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        r = np.random.random(chromo.parameters['W' + str(layer)].shape)
        child.parameters['W' + str(layer)] = pmu_children_generate(chromo.parameters['W' + str(layer)], r,
                                                                   mf_mutationratio, mf_polynomialmutationindex)
        r = np.random.random(chromo.parameters['b' + str(layer)].shape)
        child.parameters['b' + str(layer)] = pmu_children_generate(chromo.parameters['b' + str(layer)], r,
                                                                   mf_mutationratio, mf_polynomialmutationindex)


def op_uniformcrossover(chromo1, chromo2):
    L = len(mfeatask.TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        r = np.random.random(chromo1.parameters['W' + str(layer)].shape)
        t = chromo1.parameters['W' + str(layer)][r > 0.5]
        chromo1.parameters['W' + str(layer)][r > 0.5] = chromo2.parameters['W' + str(layer)][r > 0.5]
        chromo2.parameters['W' + str(layer)][r > 0.5] = t

        r = np.random.random(chromo1.parameters['b' + str(layer)].shape)
        t = chromo1.parameters['b' + str(layer)][r > 0.5]
        chromo1.parameters['b' + str(layer)][r > 0.5] = chromo2.parameters['b' + str(layer)][r > 0.5]
        chromo2.parameters['b' + str(layer)][r > 0.5] = t



def test_op_crossover():
    np.random.seed(1)
    idv1 = Chromosome()
    idv1.initialize_parameters()

    idv2 = Chromosome()
    idv2.initialize_parameters()

    (child1, child2) = op_crossover(idv1, idv2, 3.0)

    L = len(mfeatask.TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        print('weight check', layer, (idv1.parameters['W' + str(layer)] + idv2.parameters['W' + str(layer)]) - (
            child1.parameters['W' + str(layer)] + child2.parameters['W' + str(layer)]))
        print('biases check', layer, (idv1.parameters['b' + str(layer)] + idv2.parameters['b' + str(layer)]) - (
            child1.parameters['b' + str(layer)] + child2.parameters['b' + str(layer)]))


def test_op_mutate():
    np.random.seed(1)
    idv = Chromosome()
    idv.initialize_parameters()

    child = op_mutate(idv, 1.0, 0.5)

    idv.print_parameters()
    child.print_parameters()

    L = len(mfeatask.TASK_MAX)
    # parallelizable
    for layer in range(1, L):
        print('weight diff', layer, idv.parameters['W' + str(layer)] - child.parameters['W' + str(layer)])
        print('biases diff', layer, idv.parameters['b' + str(layer)] - child.parameters['b' + str(layer)])


def sigmoid_forward(x):
    return 1.0 / (1.0 + np.exp(-x)), x


def tanh_foward(x):
    return np.tanh(x), x


def relu_forward(x):
    return np.maximum(0, x), x


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + (b)  # W : (0, 1) => (-5, 5)
    # b : (0, 1) => (-1, 1)
    ### END CODE HERE ###

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid_forward(Z)
        ### END CODE HERE ###
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_forward(Z)
        ### END CODE HERE ###
    elif activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh_foward(Z)
        ### END CODE HERE ###

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
        ### END CODE HERE ###

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    ### END CODE HERE ###

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    #cost = -1.0 / m * np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))
    cost = 0.5 * np.mean((Y - AL) ** 2) # equivalent: 1.0 / (2 * m) * np.sum(np.power(Y - AL, 2))
    ### END CODE HERE ###

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def sigmoid_backward(dA, activation_cache):
    A = activation_cache
    dZ = dA * (A * (1 - A))
    return dZ


def relu_backward(dA, activation_cache):
    A = activation_cache
    dZ = dA * (A > 0)
    return dZ


def tanh_backward(dA, activation_cache):
    A = activation_cache
    dZ = dA * (1 - A ** 2)
    return dZ


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = 1.0 / m * np.dot(dZ, A_prev.T)
    db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    elif activation == "tanh":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in (range(1, L + 1)):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    ### END CODE HERE ###
    return parameters


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL
    ### END CODE HERE ###

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid")
    ### END CODE HERE ###

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def test_forward_propagation():

    inputHandler = InputHandler()
    X_train, Y_train = inputHandler.ionosphere(link=mfeatask.DATASET_IONOSPHERE)

    mfeatask.TRAINING_SIZE = X_train.shape[1]
    mfeatask.TESTING_SIZE = X_train.shape[1]

    mfeatask.NUMBEROF_INPUT = X_train.shape[0]
    mfeatask.NUMBEROF_OUTPUT = Y_train.shape[0]

    mfeatask.redefineTasks()


    idv = Chromosome()
    idv.initialize_parameters()
    AL, caches = L_model_forward(X_train, idv.parameters)
    AL2 = idv.forward_eval(1, X_train, Y_train)


    for i in range(100):
        idv = Chromosome()
        idv.initialize_parameters()
        temp = idv.forward_eval(1, X_train, Y_train)
        print(temp)

    print('AL = ', AL)
    print(AL > 0.5)
    print(compute_cost(AL, Y_train))

    parameters = L_layer_model(X_train, Y_train, mfeatask.TASK_MAX, num_iterations=2500, learning_rate=0.5, print_cost=True)

def test_idv_eval(n):
    inputHandler = InputHandler()
    X_train, Y_train = inputHandler.ionosphere(link=mfeatask.DATASET_IONOSPHERE)
    mfeatask.TRAINING_SIZE = X_train.shape[1]
    mfeatask.TESTING_SIZE = X_train.shape[1]

    mfeatask.NUMBEROF_INPUT = X_train.shape[0]
    mfeatask.NUMBEROF_OUTPUT = Y_train.shape[0]

    mfeatask.redefineTasks()
    for i in range(n):
        idv = Chromosome()
        idv.initialize_parameters()
        temp = idv.forward_eval(mfeatask.TASKS_LAYERSIZE[2], mfeatask.TASKS[2], X_train, Y_train)
        print(temp)