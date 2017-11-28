import numpy as np
def generateNbitDataSet(N):
    dim = [int(N), int(2**N)]
    sample = np.random.rand(dim[0], dim[1])
    X = np.empty(dim, dtype=bool)
    Y = np.empty([1, dim[1]], dtype=bool)

    for i in range(0, len(sample)):
        for j in range(0, len(sample[i])):
            if sample[i][j] > 0.5:
                X[i][j] = 1
            else:
                X[i][j] = 0

    for j in range(0, len(X[i])):
        count = np.count_nonzero(X[::1, j])
        #print(count)
        if count % 2 == 0: 
            Y[0][j] = 0
        else:
            Y[0][j] = 1

    print('X')
    print(X.astype(float))
    print('Y')
    print(Y.astype(float))
    return X, Y
    pass
    
def main():
    N = 7
    generateNbitDataSet(N)
    pass
if __name__ == '__main__':
    main()