import numpy as np

def split_data(X, y, test_size, shuffle=True):
    if shuffle == True:
        ids_shuffle = np.array(range(X.shape[0]))
        np.random.shuffle(ids_shuffle)

        X = X[ids_shuffle]
        y = y[ids_shuffle]

    size = int(X.shape[0]*test_size)

    X_train = X[:X.shape[0] - size]
    X_test = X[X.shape[0] - size:]
    y_train = y[:y.shape[0] - size]
    y_test = y[y.shape[0] - size:]

    return X_train, y_train, X_test, y_test

def shuffle_data(X, y):
    ids_shuffle = np.array(range(X.shape[0]))
    np.random.shuffle(ids_shuffle)

    X_ = X[ids_shuffle]
    y_ = y[ids_shuffle]

    return X_, y_

if __name__ == '__main__':
   a = np.array(range(1, 101))
   b = np.array(range(101, 201))

   X_train, y_train, X_test, y_test = split_data(a, b, 0.2)

   print (X_train)
   print (y_train)
   print (X_test)
   print (y_test)