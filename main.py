from utils import data_loader
from natural_language_processing import feature_extraction
import os
import numpy as np
np.set_printoptions(threshold=np.nan)

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

if __name__ == '__main__':

    #Train data
    data_loader = data_loader.DataLoader(DIR_PATH + 'data/data_train/')
    contents_train, labels_train = data_loader.get_data_and_label()

    #Load dictionary
    feature_extraction = feature_extraction.FeatureExtraction(None, 'dictionary.txt')
    feature_extraction.data = contents_train
    feature_extraction.load_dictionary()

    X_train = np.array(feature_extraction.bag_of_words(contents_train))
    labels_train = np.array(labels_train)

    print(X_train.shape)
    print(labels_train.shape)

    #Test data
    data_loader.data_path = DIR_PATH + 'data/data_test/'
    contents_test, labels_test = data_loader.get_data_and_label()
    X_test = np.array(feature_extraction.bag_of_words(contents_test))
    labels_test = np.array(labels_test)

    print(X_test.shape)
    print(labels_test.shape)