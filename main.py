import timeit

import numpy as np

import MFEATask as mfeatask
from Chromosome import *
from MFEA import *
from InputHandler import *


def prepareDataSet():
    inputHandler = InputHandler()
    X_data, Y_data = inputHandler.generateNbitDataSet(8)#ticTacToe(link=mfeatask.DATASET_TICTACTOE)
    m = X_data.shape[1] # number of samples

    train_ratio = 0.7

    X_train = X_data[:, :int(train_ratio * m)]
    Y_train = Y_data[:, :int(train_ratio * m)]
    X_test = X_data[:, int(train_ratio * m):]
    Y_test = Y_data[:, int(train_ratio * m):]

    mfeatask.TRAINING_SIZE = X_train.shape[1]
    mfeatask.TESTING_SIZE = X_train.shape[1]

    mfeatask.NUMBEROF_INPUT = X_train.shape[0]
    mfeatask.NUMBEROF_OUTPUT = Y_train.shape[0]

    mfeatask.redefineTasks()

    return X_train, Y_train, X_test, Y_test

def main():
    #test_forward_propagation()
    X_train, Y_train, X_test, Y_test = prepareDataSet()

    print('number of tasks =', mfeatask.NUMBEROF_TASKS)
    print('number of layers =', mfeatask.NUMBEROF_LAYERS)
    print(mfeatask.TASKS_LAYERSIZE)
    print(mfeatask.TASKS)
    print(mfeatask.TASK_MAX)

    mfea = MFEA(X_train, Y_train, X_test, Y_test, 90, 1000)
    mfea.evolution()
    mfea.sumarizeTrainingStep()
    mfea.revalAccuracyOnTestingData()




if __name__ == "__main__":
    main()