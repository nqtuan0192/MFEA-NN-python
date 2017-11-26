import timeit
import numpy as np
import MFEATask as gv
from Chromosome import *
from MFEA import *


def my_function():
    print('donothing')

def my_function_two():
    print('dosomething')

def main():
    print('number of tasks =', gv.NUMBEROF_TASKS)
    print('number of layers =', gv.NUMBEROF_LAYERS)
    print(gv.TASKS_LAYERSIZE)
    print(gv.TASKS)
    print(gv.TASK_MAX)


    test_forward_propagation()




if __name__ == "__main__":
    main()