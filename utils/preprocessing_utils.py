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

if __name__ == '__main__':
   a = np.array(range(1, 101))
   b = np.array(range(101, 201))

   X_train, y_train, X_test, y_test = split_data(a, b, 0.2)

   print (X_train.shape)
   print (y_train.shape)
   print (X_test.shape)
   print (y_test.shape)