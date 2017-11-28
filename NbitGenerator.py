import numpy as np
def generateNbitDataSet(N):
    dim = [int(2**N), int(N)]
    sample = np.random.rand(dim[0], dim[1])
    X = np.zeros(dim, dtype=bool)
    Y = np.empty([1, dim[0]], dtype=bool)

    for i in range(0, len(X)):
        increment(X[i], i + 1)
        #print(X[i].astype(float))
    
    #X = X.T
    np.random.shuffle(X)
    # print(X)
    for i in range(0, len(X)):
        for j in range(0, len(X[i])):
            count = np.count_nonzero(X[::1, j])
            #print(count)
            if count % 2 == 0: 
                Y[0][j] = 0
            else:
                Y[0][j] = 1

    # print('X')
    # print(X.astype(float).T)
    # print('Y')
    # print(Y.astype(float))
    return X.astype(float).T, Y.astype(float)
    pass


def increment(X, times):
    cur_times = 0
    while cur_times != times:
        i = 0
        while i < len(X) and X[i] == 1:
            X[i] = 0
            i = i + 1
        if i < len(X):
            X[i] = 1
        cur_times += 1
    
def main():
    N = 3
    generateNbitDataSet(N)
    pass


if __name__ == '__main__':
    main()
