# Basic configuration file containing paths.

class Config(object):
  DEBUG = False
  DEVELOPMENT = False
  UPLOAD_FOLDER = '/home/lehel/data/wav'
  USER_DICT_PATH = '/home/lehel/data/wav/user_data.dict'
  PREPROCESSED_PATH = '/home/lehel/data/preprocessed_audio'
  IDENTIFY_PATH = '/home/lehel/data/identify'
  SIAMESE_MODEL_PATH = '/home/lehel/model/siamese__nseconds_3.0__filters_32__embed_64__drop_0.05__r_0.hdf5'
  ALLOWED_EXTENSIONS = set(['wav'])

class ProductionConfig(Config):
  pass

class DevelopmentConfig(Config):
  DEBUG = True

class TestingConfig(Config):
  TESTING = True
