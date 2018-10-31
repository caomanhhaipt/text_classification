import configparser
import os

class Config(object):
    def __init__(self, path):
        self.path = path
        self.config_file = configparser.ConfigParser()

    def create_config(self):
        config = configparser.ConfigParser()

        config['nlp'] = {'no_below': 2, 'no_above': 0.15}
        config['labels'] = {'chinh_tri_xa_hoi':1, 'doi_song':2, 'khoa_hoc':3, 'kinh_doanh':4, 'phap_luat':5,
          'suc_khoe':6, 'the_gioi':7, 'the_thao':8, 'van_hoa':9, 'vi_tinh':10}

        config.write(open(self.path, 'w'))

    def get_config_file(self):
        if not os.path.exists(self.path):
            self.create_config()

        self.config_file.read(self.path)

    def get_setting(self, section, name_setting):
        return self.config_file.get(section, name_setting)

    def update_setting(self, section, name_setting, value):
        self.config_file.set(section, name_setting, value)

        with open(self.path, "wb") as config:
            self.config_file.write(config)

    def delete_setting(self, section, name_setting):
        self.config_file.remove_option(section, name_setting)

        with open(self.path, "wb") as config:
            self.config_file.write(config)

if __name__ == "__main__":
    path = "setting.ini"

    config = Config(path)
    config.get_config_file()
    print (float(config.get_setting("natural_language_processing", "special_character")))