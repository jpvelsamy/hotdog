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
        self.features = myfeatures.split(",")
        self.model_save_path = self.config_object.get_model_save_path() + '/ace_cpl.h5'

    def restore_model(self):
        self.model = keras.models.load_model(self.model_save_path)

    def test(self, input_file):
        data = pd.read_csv(input_file, engine='c', dtype='float64', names=self.features, header=0, skiprows=0)
        mean = data.mean(axis=0)
        data -= mean
        std = data.std(axis=0)
        data /= std
        outcome = self.model.predict(data)
        df_out = pd.merge(data, outcome, how='left', left_index=True, right_index=True)
        logger.info(df_out.head(10))
        df_out.to_csv(self.outcome_file + '/test_outcome.csv', float_format='%.2f')