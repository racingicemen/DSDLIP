import numpy as np
import pandas as pd

# normalize numerical column
# one-hot encoded categorical column

def get_data():

    df = pd.read_csv('ecommerce_data.csv')

    # convert this into a numpy array
    # shape of data is (500,6), which is 500 rows of data (excluding header row) with 6 columns each
    data = df.as_matrix()

    # X is all rows, and all columns except the last one
    X = data[:, :-1]
    # Y is all rows and just the last column
    Y = data[:, -1]

    # normalize columns 1 and 2
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))

    # 0:D-1 means 0,1,2,...,D-2
    # columns 0 through D-2 in X2 are identical to those in X
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    # new columns in X2 are D-1, D, D+1, D+2 filled by one-hot encoding
    for n in range(N):
        t = int(X[n,D-1])
        X2[n, t+D-1] = 1

    """
    alternative approach to implement one-hot encoding
    Z = np.zeros((N, 4))
    Z[np.range(N), X[:,D-1].astype(np.int32)] = 1
    X2[:,-4:] = Z
    """

    return X2, Y

def get_binary_data():
    # Y can be 0,1,2,3 i.e. data is in 4 classes. get_binary_data returns data from the first two classes
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

if __name__ == "__main__":
    X, Y = get_binary_data()
    print(X.shape, Y.shape)
