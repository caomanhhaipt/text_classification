import os
from utils.file_reader import FileReader
from natural_language_processing.nlp import NLP
from config import config_parser

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"
con = config_parser.Config(DIR_PATH + "config/settings.ini")
con.get_config_file()

class DataLoader(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def get_files(self):
        folders = []
        for folder in os.listdir(self.data_path):
            folders.append(self.data_path + folder + '/')

        titles = os.listdir(self.data_path)
        files = {}
        for folder, title in zip(folders, titles):
            tmp_files = []
            for file in os.listdir(folder):
                tmp_files.append(folder + file)
            files[title] = tmp_files

        self.files = files

    def get_data_and_label(self):
        self.get_files()
        nlp = NLP(None)
        nlp.set_stop_words()
        file_reader = FileReader(None)
        contents = []
        labels = []

        #init titles
        titles = {}
        for title in self.files:
            titles[title] = con.get_setting('labels', str(title))

        for title in self.files:
            for file in self.files[title]:
                file_reader.file_path = file
                nlp.text = file_reader.read()

                contents.append(nlp.get_words_feature())
                labels.append(titles[title])

        return (contents, labels)

if __name__ == '__main__':
    loader = DataLoader('/home/haicm/text_classfication/data/test/')
    loader.get_data_and_label()