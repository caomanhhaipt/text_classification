import numpy as np

class MultiSVM(object):
    def __init__(self, X_train, y_train, batch_size=1000, num_iters=50, print_every=10):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.print_every = print_every

    def bias_trick(self, C=10):
        self.X_train = np.concatenate((self.X_train, np.ones((self.X_train.shape[0], 1))), axis = 1)
        self.W_init = 0.00001 * np.random.randn(self.X_train.shape[1], C)

    def svm_loss(self, W, X, y, reg):
        N = X.shape[0]

        #step 1
        Z = X.dot(W)
        # print (Z)
        #step 2
        id0 = np.arange(Z.shape[0])
        # print (id0)
        # print (y)
        correct_class_score = Z[id0, y].reshape(N, 1)
        margins = np.maximum(0, Z - correct_class_score + 1)
        margins[id0, y] = 0

        #step 3
        F = (margins > 0).astype(int)
        F[np.arange(F.shape[0]), y] = np.sum(-F, axis=1)

        loss = np.sum(margins)
        loss /= N
        loss += 0.5 * reg * np.sum(W * W)

        dW = X.T.dot(F) / N + reg * W
        return loss, dW

    def build_model(self, reg=0.1, lr=0.1, path=None):
        print (path)
        self.bias_trick()
        self.W = self.W_init
        loss_history = []

        for it in range(self.num_iters):
            ids_shuffle = np.array(range(self.X_train.shape[0]))
            np.random.shuffle(ids_shuffle)

            n_batches = int(np.ceil(self.X_train.shape[0] / float(self.batch_size)))

            for ib in range(n_batches):
                last_id = min(self.batch_size*(ib + 1), self.X_train.shape[0])
                ids = ids_shuffle[self.batch_size * ib: last_id]
                X_batch = self.X_train[ids]
                y_batch = self.y_train[ids]
                lossib, dw = self.svm_loss(self.W, X_batch, y_batch, reg)
                loss_history.append(lossib)
                self.W -= lr * dw

            if it % self.print_every == 0 and it > 0:
                with open(path + "test.log", "a") as m_file:
                    m_file.write('it % d / % d, loss = % f' % (it, self.num_iters, loss_history[it]))
                    m_file.write('\n')
                # print('it % d / % d, loss = % f' % (it, self.num_iters, loss_history[it]))

        return loss_history

    def predict(self, X):
        Z = X.dot(self.W)

        return np.argmax(Z, axis=1)

    def evaluate(self, X, y, path=None):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y_pred = self.predict(X)
        # print (100*np.mean(y_pred == y))
        i = 0
        count_true = 0
        for item in y:
            if item == y_pred[i]:
                count_true += 1
            i += 1

        with open(path + "test.log", "a") as m_file:
            m_file.write("True/Total: " + str(count_true) + "/" + str(len(y)))
            m_file.write('\n')
        # print ("True/Total: " + str(count_true) + "/" + str(len(y)))

    def save_weight_as_txt(self, path):
        np.savetxt(path, self.W)