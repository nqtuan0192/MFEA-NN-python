import timeit

import numpy as np

import MFEATask as mfeatask
from Chromosome import *
from MFEA import *
from InputHandler import *


def my_function():
    print('donothing')

def my_function_two():
    print('dosomething')

def main():
    #test_forward_propagation()

    inputHandler = InputHandler()
    X_train, Y_train = inputHandler.ionosphere(link=mfeatask.DATASET_IONOSPHERE)
    mfeatask.TRAINING_SIZE = X_train.shape[1]
    mfeatask.TESTING_SIZE = X_train.shape[1]

    mfeatask.NUMBEROF_INPUT = X_train.shape[0]
    mfeatask.NUMBEROF_OUTPUT = Y_train.shape[0]

    mfeatask.redefineTasks()

    print('number of tasks =', mfeatask.NUMBEROF_TASKS)
    print('number of layers =', mfeatask.NUMBEROF_LAYERS)
    print(mfeatask.TASKS_LAYERSIZE)
    print(mfeatask.TASKS)
    print(mfeatask.TASK_MAX)

    mfea = MFEA(X_train, Y_train, 90, 1000)
    mfea.evolution()




if __name__ == "__main__":
    main()