import logging

import pandas as pd
import numpy as np
from pandas import DataFrame
from tensorflow import keras


from hotdogconfig import Configuration

logger = logging.getLogger("ACE")

class AskJunoAPI:
    def __init__(self, config_object: Configuration):
        self.config_object = config_object
        # self.features = ['reach','impressions','results','amount','frequency','clicks','cpc','ctr','cpreach','cpm','engagement']
        eval_features_config = self.config_object.get_cpl_features()
        self.eval_features = eval_features_config.split(",")
        test_features_config = self.config_object.get_test_feature_names()
        self.test_features = test_features_config.split(",")
        self.model_save_path = self.config_object.get_model_save_path() + '/ace_cpl.h5'
        self.outcome_file = self.config_object.get_cpl_outcome()

    # https://www.tensorflow.org/guide/keras/save_and_serialize
    def restore_model(self):
        self.model = keras.models.load_model(self.model_save_path)

    def test(self, sigma_folder):
        source_file_as_sigma = sigma_folder + self.config_object.get_path_separator() + self.config_object.get_input_file_name()
        source_data:DataFrame = pd.read_csv(source_file_as_sigma, engine='c', dtype='float64', names=self.test_features, header=0,
                                  skiprows=0)

        cpl_data_frame:DataFrame = pd.read_csv(source_file_as_sigma, engine='c', dtype='float64', names=['cpr'], header=0,
                                  skiprows=0)

        cpl_mean = cpl_data_frame.mean(axis=0)
        cpl_std = cpl_data_frame.std(axis=0)

        mean = source_data.mean(axis=0)
        std = source_data.std(axis=0)

        test_file_as_gamma = sigma_folder + self.config_object.get_path_separator() + self.config_object.get_test_file_name()
        test_data:DataFrame = pd.read_csv(test_file_as_gamma, engine='c', dtype='float64', names=self.test_features, header=0,
                                skiprows=0)

        new_test_data:DataFrame = pd.read_csv(test_file_as_gamma, engine='c', dtype='float64', names=self.test_features, header=0,
                                skiprows=0)

        new_test_data -= mean
        new_test_data /= std

        logger.info(f'inbound test_data # {test_data.head}')
        outcome = self.model.predict(new_test_data)
        logger.info(f'outcome before synthesizing# {outcome}')
        index = ['Row_' + str(i)
                 for i in range(1, len(outcome) + 1)]

        # defining column headers for the
        # Pandas dataframe
        columns = ['Column_' + str(i)
                   for i in range(1, len(outcome[0]) + 1)]

        #np.multiply(outcome, cpl_std.get(key='cpr'))
        #np.add(outcome, cpl_mean.get(key='cpr'))

        df_outcome = pd.DataFrame(outcome, index=index, columns=columns)
        df_out = pd.merge(test_data, df_outcome, how='left', left_index=True, right_index=True)
        df_out.to_csv(
            self.outcome_file + self.config_object.get_path_separator() + self.config_object.get_test_file_name() + '_outcome.csv',
            float_format='%.2f')
