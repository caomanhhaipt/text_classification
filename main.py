from utils import data_loader
from natural_language_processing import feature_extraction
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
from model import multi_SVM
from utils import preprocessing_utils

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

if __name__ == '__main__':

    #Train data
    data_loader = data_loader.DataLoader(DIR_PATH + 'data/data_train/')
    contents_train, labels_train = data_loader.get_data_and_label()

    #Load dictionary
    feature_extraction = feature_extraction.FeatureExtraction(contents_train, DIR_PATH + 'config/dictionary.txt')
    feature_extraction.load_dictionary()

    #bag of words
    X_train = np.array(feature_extraction.bag_of_words(contents_train))
    labels_train = np.array(labels_train).astype(int)

    #tf_idf
    # X_train = np.array(feature_extraction.tf_idf(contents_train))
    # y_train = np.array(labels_train).astype(int)

    #split data
    X_train, y_train, X_val, y_val = preprocessing_utils.split_data(X_train, labels_train, 0.2)
    with open(DIR_PATH + "test.log", "a") as m_file:
        m_file.write('Train data:')
        m_file.write(str(X_train.shape))
        m_file.write(str(y_train.shape))
        m_file.write('\n')
    # print(X_train.shape)
    # print(y_train.shape)

    # Validation data
    # data_loader.data_path = DIR_PATH + 'data/data_validation/'
    # contents_val, labels_val = data_loader.get_data_and_label()

    # bag of words
    # X_val = np.array(feature_extraction.bag_of_words(contents_val))

    # tf_idf
    # X_val = np.array(feature_extraction.tf_idf(contents_val))
    # y_val = np.array(labels_val).astype(int)

    # print (X_val.shape)
    # print (y_val.shape)
    with open(DIR_PATH + "test.log", "a") as m_file:
        m_file.write('Validation data:')
        m_file.write(str(X_val.shape))
        m_file.write(str(y_val.shape))
        m_file.write('\n')
    #Test data
    data_loader.data_path = DIR_PATH + 'data/data_test/'
    contents_test, labels_test = data_loader.get_data_and_label()

    #bag of words
    X_test = np.array(feature_extraction.bag_of_words(contents_test))

    #tf_idf
    # X_test = np.array(feature_extraction.tf_idf(contents_test))
    y_test = np.array(labels_test).astype(int)

    with open(DIR_PATH + "test.log", "a") as m_file:
        m_file.write('Test data:')
        m_file.write(str(X_test.shape))
        m_file.write(str(y_test.shape))
        m_file.write('\n')

    #Train model
    model = multi_SVM.MultiSVM(X_train=X_train, y_train=y_train)
    model.build_model(path=DIR_PATH)
    model.evaluate(X_test, y_test, DIR_PATH)
    model.evaluate(X_val, y_val, DIR_PATH)
    model.save_weight_as_txt('w.txt')