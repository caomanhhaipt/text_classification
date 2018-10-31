import os
import numpy as np
from config import config_parser

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"
con = config_parser.Config(DIR_PATH + "config/settings.ini")
con.get_config_file()

NO_BELOW = con.get_setting('nlp', 'no_below')
NO_ABOVE = con.get_setting('nlp', 'no_above')

class FeatureExtraction(object):
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def filter_extremes(self, no_below, no_above):
        total_text = len(self.data)
        item_remove = []
        for item in self.dict:
            if self.dict[item] < NO_BELOW or (self.dict[item]*1.0)/total_text > NO_ABOVE:
                item_remove.append(item)

        for item in item_remove:
            print (item)
            self.dict.pop(item, None)

    def build_dictionary(self):
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

        self.filter_extremes(no_below, no_above)
        i = 0
        for item in self.dict:
            tmp = str(i) + '\t' + str(item) + '\t' + str(self.dict[item]) + '\n'
            with open(self.path, 'a') as file:
                file.write(tmp)
            i += 1

    def load_dictionary(self):
        if not os.path.exists(self.path):
            self.build_dictionary()

        with open(self.path, 'r') as file:
            a = file.read().split('\n')

        self.len_dict = 0
        self.dict = {}
        for item in a:
            tmp = item.split('\t')
            if tmp != ['']:
                self.len_dict += 1
                self.dict[tmp[1]] = tmp[0]

    def bag_of_words(self, contents):
        X_train = []
        for words in contents:
            tmp = np.zeros(self.len_dict).astype(int)
            for word in words:
                if word in self.dict:
                    # print (word)
                    tmp[int(self.dict[word])] += 1
            X_train.append(tmp)

        return X_train