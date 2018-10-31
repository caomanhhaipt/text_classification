class FileReader(object):
    def __init__(self, file_path, encoder=None):
        self.file_path = file_path
        self.encoder = encoder if encoder != None else 'utf-16le'

    def read(self):
        with open(self.file_path, encoding=self.encoder) as f:
            content = f.read()

        return content

    def read_stopwords(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            tmp_stop_words = []
            for w in f.readlines():
                tmp_stop_words.append(w.strip().replace(' ', '_'))

            stopwords = set(tmp_stop_words)
        return stopwords

if __name__ == '__main__':
    file = FileReader('/home/haicm/text_classfication/config/stop_words.txt')
    content = file.read_stopwords()

    print (content)