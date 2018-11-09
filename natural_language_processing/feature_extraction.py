import os
import numpy as np
from config import config_parser
import math

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"
con = config_parser.Config(DIR_PATH + "config/settings.ini")
con.get_config_file()

NO_BELOW = con.get_setting('nlp', 'no_below')
NO_ABOVE = con.get_setting('nlp', 'no_above')

class FeatureExtraction(object):
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def filter_extremes(self):
        total_text = len(self.data)
        item_remove = []
        for item in self.dict:
            if self.dict[item] < float(NO_BELOW) or (self.dict[item]*1.0)/total_text > float(NO_ABOVE):
                item_remove.append(item)

        for item in item_remove:
            # print (item)
            self.dict.pop(item, None)

    def build_dictionary(self):
        print ('building dictionary')
        self.dict = {}
        for words in self.data:
            for word in words:
                if word not in self.dict:
                    self.dict[word] = 0

        for words in self.data:
            tmp_dict = {}
            for word in words:
                if word not in tmp_dict:
                    tmp_dict[word] = 0

            for item in tmp_dict:
                self.dict[item] += 1

        self.filter_extremes()
        i = 0
        for item in self.dict:
            tmp = str(i) + '\t' + str(item) + '\t' + str(self.dict[item]) + '\n'
            with open(self.path, 'a') as file:
                file.write(tmp)
            i += 1

        print('finish build dictionary')

    def load_dictionary(self):
        if os.path.exists(self.path):
            os.remove(self.path)

        self.build_dictionary()

        with open(self.path, 'r') as file:
            a = file.read().split('\n')

        self.len_dict = 0
        self.dict = {}
        self.dict_number = {}
        for item in a:
            tmp = item.split('\t')
            if tmp != ['']:
                self.len_dict += 1
                self.dict[tmp[1]] = tmp[0]
                self.dict_number[tmp[1]] = tmp[2]

        print ('finish load dictionary')

    def bag_of_words(self, contents):
        # print (self.dict)
        X = []
        # print (contents)
        for words in contents:
            tmp = np.zeros(self.len_dict).astype(float)
            for word in words:
                if word in self.dict:
                    # print (word)
                    tmp[int(self.dict[word])] += 1.0
            X.append(tmp)

        return X

    def mini_batch_bag_of_words(self, contents, batch_size):
        n_batches = int(np.ceil(np.array(contents).shape[0] / float(batch_size)))

        X = None
        for ib in range(n_batches):
            # print(ib)
            last_id = min(batch_size * (ib + 1), np.array(contents).shape[0])
            contents_batch = contents[batch_size * ib: last_id]
            tmp = np.array(self.bag_of_words(contents_batch))
            if ib == 0:
                X = tmp
            else:
                X = np.concatenate((X, tmp), axis=0)

        return X

    def tf_idf(self, contents):
        X = []
        a =  len(contents)
        i = 0
        for words in contents:
            print (str(i) + '/' + str(a))
            tmp = np.zeros(self.len_dict).astype(float)

            for key in self.dict:
                #tf
                f = len(np.where(np.array(words)==key)[0])
                if f == 0:
                    continue

                tf = 1.0*f/len(words)

                #idf
                # count = 0
                # for item in contents:
                #     if len(np.where(np.array(item)==key)[0]) != 0:
                #         count += 1
                # print (str(count) + ':' + str(self.dict_number[key]))
                idf = math.log10(1.0*len(self.data)/float(self.dict_number[key]))

                #tf_idf
                tmp[int(self.dict[key])] = tf*idf

            X.append(tmp)

            i += 1

        return X

    def mini_batch_tf_idf(self, contents, batch_size):
        print ("-------------------------------------------------")
        n_batches = int(np.ceil(np.array(contents).shape[0] / float(batch_size)))

        X = None
        for ib in range(n_batches):
            # print(ib)
            last_id = min(batch_size * (ib + 1), np.array(contents).shape[0])
            contents_batch = contents[batch_size * ib: last_id]
            tmp = np.array(self.tf_idf(contents_batch))
            if ib == 0:
                X = tmp
            else:
                X = np.concatenate((X, tmp), axis=0)

        return X