import numpy as np

NONE_LAYER = 0

TRAINING_SIZE = 256
TESTING_SIZE = 256

NUMBEROF_INPUT = 2
NUMBEROF_OUTPUT = 1


TASKS_LAYERSIZE = np.array([2, 3, 4])
TASKS = np.array([[NUMBEROF_INPUT, 4, NUMBEROF_OUTPUT, NONE_LAYER, NONE_LAYER],
                  [NUMBEROF_INPUT, 3, 1, NUMBEROF_OUTPUT, NONE_LAYER],
                  [NUMBEROF_INPUT, 6, 3, 1, NUMBEROF_OUTPUT]])

NUMBEROF_TASKS = len(TASKS)
NUMBEROF_LAYERS = max(len(TASKS[i]) - 1 for i in range(len(TASKS)))

TASK_MAX = [max(TASKS[:, layer]) for layer in range(NUMBEROF_LAYERS + 1)]