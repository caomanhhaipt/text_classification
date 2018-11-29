from pyvi import ViTokenizer
from utils import file_reader
import os

SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''
DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"
class NLP(object):
    def __init__(self, text):
        self.text = text

    def segmentation(self):
        # print (ViTokenizer.tokenize(self.text))
        return ViTokenizer.tokenize(self.text)

    def set_stop_words(self):
        self.stop_words = file_reader.FileReader(DIR_PATH + 'config/stop_words.txt').read_stopwords()

    def split_words(self):
        text = self.segmentation()

        tmp = []
        try:
            for word in text.split():
                tmp.append(word.strip(SPECIAL_CHARACTER).lower())
        except:
            pass

        return tmp

    def get_words_feature(self):
        split_words = self.split_words()

        tmp_words = []
        for word in split_words:
            if word not in self.stop_words:
                tmp_words.append(word)

        words = []
        for word in tmp_words:
            if word == '\ufeff' or word == '':
                continue
            words.append(word)
        return words

if __name__ == '__main__':
    # file = file_reader.FileReader('/home/haicm/text_classfication/data/data_train/chinh_tri_xa_hoi/XH_NLD_ (3672).txt')
    # content = file.read()
    # print (content)
    print (u"Đại học, bách khoa hà nội 'Sinh viên đại học'")
    nlp1 = NLP(u"Đại học, bách khoa hà nội 'Sinh viên đại học'")
    print (nlp1.segmentation())
    # nlp1.set_stop_words()
    # # print (nlp1.stop_words)
    # # print (nlp1.segmentation())
    # # #
    # print (nlp1.get_words_feature())