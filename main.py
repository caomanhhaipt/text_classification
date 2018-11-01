from utils import data_loader
from natural_language_processing import feature_extraction
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
from model import linear_svr

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

if __name__ == '__main__':

    #Train data
    data_loader = data_loader.DataLoader(DIR_PATH + 'data/data_train/')
    contents_train, labels_train = data_loader.get_data_and_label()

    #Load dictionary
    feature_extraction = feature_extraction.FeatureExtraction(contents_train, 'dictionary.txt')
    feature_extraction.load_dictionary()

    #bag of words
    X_train = np.array(feature_extraction.bag_of_words(contents_train))

    #tf_idf
    # X_train = np.array(feature_extraction.tf_idf(contents_train))
    labels_train = np.array(labels_train)

    print(X_train.shape)
    print(labels_train.shape)

    #Test data
    data_loader.data_path = DIR_PATH + 'data/data_test/'
    contents_test, labels_test = data_loader.get_data_and_label()

    #bag of words
    X_test = np.array(feature_extraction.bag_of_words(contents_test))

    #tf_idf
    #X_test = np.array(feature_extraction.tf_idf(contents_test))
    labels_test = np.array(labels_test)

    print(X_test.shape)
    print(labels_test.shape)

    #Train model
    est = linear_svr.Classifier(features_train=X_train, features_test=X_test, labels_train=labels_train,
                     labels_test=labels_test)
    est.training()