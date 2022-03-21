import logging

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


logger = logging.getLogger("ACE")

class AskJunoAPI:
    def __init__(self, config_object):
        self.config_object = config_object
        myfeatures = self.config_object.get_cpl_features()
        self.features = ['reach','impressions','results','amount','frequency','clicks','cpc','ctr','cpreach','cpm','engagement']
        self.model_save_path = self.config_object.get_model_save_path() + '/ace_cpl.h5'
        self.outcome_file = self.config_object.get_cpl_outcome()

    # https://www.tensorflow.org/guide/keras/save_and_serialize
    def restore_model(self):
        self.model = keras.models.load_model(self.model_save_path)

    def test(self, input_file):
        data = pd.read_csv(input_file, engine='c', dtype='float64', names=self.features, header=0, skiprows=0)
        mean = data.mean(axis=0)
        data -= mean
        std = data.std(axis=0)
        data /= std
        logger.info(f'inbound data # {data.head}')
        outcome = self.model.predict(data)
        logger.info(f'outcome # {outcome}')
