from utils import data_loader
from natural_language_processing import feature_extraction
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
from model import multi_SVM
from utils import preprocessing_utils
import datetime

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"

if __name__ == '__main__':
    print (datetime.datetime.now())
    #Train data
    data_loader = data_loader.DataLoader(DIR_PATH + 'data/data_train/')
    contents_train, labels_train = data_loader.get_data_and_label()

    #Load dictionary
    X_train = np.array(contents_train)
    y_train = np.array(labels_train)
    feature_extraction = feature_extraction.FeatureExtraction(X_train, DIR_PATH + 'config/dictionary.txt')
    feature_extraction.load_dictionary()

    y_train = y_train.astype(int)

    data_loader.data_path = DIR_PATH + 'data/data_validation/'
    X_val, y_val = data_loader.get_data_and_label()
    y_val = np.array(y_val).astype(int)

    # tf_idf
    X_train = feature_extraction.mini_batch_tf_idf(X_train, 1000)
    X_val = feature_extraction.mini_batch_tf_idf(X_val, 1000)

    with open(DIR_PATH + "test.log", "a") as m_file:
        m_file.write('Train data:')
        m_file.write(str(X_train.shape))
        m_file.write(str(y_train.shape))
        m_file.write('\n')

    with open(DIR_PATH + "test.log", "a") as m_file:
        m_file.write('Validation data:')
        m_file.write(str(X_val.shape))
        m_file.write(str(y_val.shape))
        m_file.write('\n')
    #Test data
    data_loader.data_path = DIR_PATH + 'data/data_test/'
    contents_test, labels_test = data_loader.get_data_and_label()

    # #tf_idf
    X_test = feature_extraction.mini_batch_tf_idf(contents_test, 1000)
    y_test = np.array(labels_test).astype(int)

    with open(DIR_PATH + "test.log", "a") as m_file:
        m_file.write('Test data:')
        m_file.write(str(X_test.shape))
        m_file.write(str(y_test.shape))
        m_file.write('\n')

    X_train, y_train = preprocessing_utils.shuffle_data(X_train, y_train)

    lr_list = [0.1, 0.075, 0.05, 0.025]
    reg_list = [1e-5, 1e-6, 1e-7]

    acc_current = 0
    loss_current = []
    # Train model
    for lr in lr_list:
        for reg in reg_list:

            with open(DIR_PATH + "test.log", "a") as m_file:
                m_file.write('----------------------------------\n')
                m_file.write("lr:" + str(lr) + "\n")
                m_file.write("reg: " + str(reg) + "\n")
                m_file.write('\n')

            # print (datetime.datetime.now())
            model = multi_SVM.MultiSVM(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, num_iters=100)
            loss = model.build_model(path=DIR_PATH, lr=lr, reg=reg)
            train_acc = model.evaluate(X_train, y_train, DIR_PATH)
            test_acc = model.evaluate(X_test, y_test, DIR_PATH)
            val_acc = model.evaluate(X_val, y_val, DIR_PATH)

            if val_acc > acc_current:
                acc_current = val_acc
                model.save_as_txt(DIR_PATH + 'w.txt')
                model.save_as_txt(DIR_PATH + 'loss.txt')

            # print (datetime.datetime.now())