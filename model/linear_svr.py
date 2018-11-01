from sklearn.svm import LinearSVC

class Classifier(object):
    def __init__(self, features_train = None, labels_train = None, features_test = None, labels_test = None,  estimator = LinearSVC(random_state=0)):
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.estimator = estimator

    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        self.training_result()

    def training_result(self):
        y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
        i = 0
        count_true = 0
        for item in y_true:
            if item == y_pred[i]:
                count_true += 1
            i += 1

        print ("True/Total: " + str(count_true) + "/" + str(len(y_true)))