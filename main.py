import numpy as np
from MFEATask import *
from Chromosome import *
from MFEA import *


def my_function():
    print('donothing')

def my_function_two():
    print('dosomething')

def main():
    print('number of tasks =', NUMBEROF_TASKS)
    print('number of layers =', NUMBEROF_LAYERS)
    print(TASKS_LAYERSIZE)
    print(TASKS)
    print(TASK_MAX)
    c = Chromosome()
    c.initialize_parameters()
    c.print_parameters()



if __name__ == "__main__":
    main()