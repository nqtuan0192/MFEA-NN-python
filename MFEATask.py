import numpy as np

DATASET_TICTACTOE = 'dataset/tic-tac-toe/tic-tac-toe.data'
DATASET_IONOSPHERE = 'dataset/ionosphere/ionosphere.data'
DATASET_CREDITSCREENING = 'dataset/credit-screening/crx.data'
DATASET_BREASTCANCER = 'dataset/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
DATASET_NBIT_INP = 'dataset/nbit/training_input'
DATASET_NBIT_OUT = 'dataset/nbit/training_output'


NONE_LAYER = 0

TRAINING_SIZE = 100
TESTING_SIZE = 10

NUMBEROF_INPUT = 2
NUMBEROF_OUTPUT = 1


TASKS_LAYERSIZE = np.array([2, 2, 2])
TASKS = np.array([[NUMBEROF_INPUT, 7, NUMBEROF_OUTPUT],
                  [NUMBEROF_INPUT, 8, NUMBEROF_OUTPUT],
                  [NUMBEROF_INPUT, 9, NUMBEROF_OUTPUT]])

NUMBEROF_TASKS = len(TASKS)
NUMBEROF_LAYERS = max(len(TASKS[i]) - 1 for i in range(len(TASKS)))

TASK_MAX = [max(TASKS[:, layer]) for layer in range(NUMBEROF_LAYERS + 1)]

def redefineTasks():
    global TRAINING_SIZE, TESTING_SIZE, NUMBEROF_INPUT, NUMBEROF_OUTPUT
    global TASKS_LAYERSIZE, TASKS, NUMBEROF_TASKS, NUMBEROF_LAYERS, TASK_MAX

    TASKS_LAYERSIZE = np.array([2, 2, 2])
    TASKS = np.array([[NUMBEROF_INPUT, 7, NUMBEROF_OUTPUT],
                      [NUMBEROF_INPUT, 8, NUMBEROF_OUTPUT],
                      [NUMBEROF_INPUT, 9, NUMBEROF_OUTPUT]])

    NUMBEROF_TASKS = len(TASKS)
    NUMBEROF_LAYERS = max(len(TASKS[i]) - 1 for i in range(len(TASKS)))

    TASK_MAX = [max(TASKS[:, layer]) for layer in range(NUMBEROF_LAYERS + 1)]