import logging
from typing import Union

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from hotdogconfig import Configuration

logger = logging.getLogger("ACE")


class AskJunoACE:
    mean: Union[Series, float]
    std: Union[Series, float]
    data: DataFrame
    x_test: DataFrame
    x_train: DataFrame
    model: None
    y_train: None
    y_test: None

    def __init__(self, config_object: Configuration, sigma_folder):
        self.ytest_std = Union[Series, float]
        self.ytest_mean = Union[Series, float]
        self.config_object = config_object
        self.sigma_folder = sigma_folder
        path_separator = self.config_object.get_path_separator()
        input_file_name = self.config_object.get_input_file_name()
        file_ext: str = ".csv"
        self.cpl_train_file = self.sigma_folder + path_separator + input_file_name + file_ext
        myfeatures = self.config_object.get_cpl_features()
        self.features = myfeatures.split(",")
        self.ratio = float(self.config_object.get_cpl_ratio())
        self.outcome_file = self.config_object.get_cpl_outcome()
        self.model_save_path = self.config_object.get_model_save_path() + '/ace_cpl.h5'
        self.epoch = int(self.config_object.get_epoch())
        self.outcome_csv_ = self.outcome_file + path_separator + input_file_name + '_outcome' + file_ext

    def load_data(self):
        self.data = pd.read_csv(self.cpl_train_file, engine='c', dtype='float64', names=self.features
                                , header=0, skiprows=0)

    def test_train_split(self):
        x = self.data.iloc[:, 0:11]
        y = self.data.iloc[:, -1]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.ratio)

    def normalize_data(self):
        # self.mean = self.data[self.features].mean(axis=0)
        # self.std = self.data[self.features].std(axis=0)

        self.train_mean = self.x_train.mean(axis=0)
        self.x_train -= self.train_mean
        self.train_std = self.x_train.std(axis=0)
        self.x_train /= self.train_std

        self.test_mean = self.x_test.mean(axis=0)
        self.x_test -= self.test_mean
        self.test_std = self.x_test.std(axis=0)
        self.x_test /= self.test_std

        # df_temp_train = pd.DataFrame()
        # df_temp_train['cpr'] = self.y_train
        # self.ytrain_mean = df_temp_train.mean(axis=0)
        # df_temp_train -= self.ytrain_mean
        # self.ytrain_std = df_temp_train.std(axis=0)
        # df_temp_train /= self.ytrain_std
        # self.y_train = df_temp_train['cpr']

        # df_temp_test = pd.DataFrame()
        # df_temp_test['cpr'] = self.y_test
        # self.ytest_mean = df_temp_test.mean(axis=0)
        # df_temp_test -= self.ytest_mean
        # self.ytest_std = df_temp_test.std(axis=0)
        # df_temp_test /= self.ytest_std
        # self.y_test = df_temp_test['cpr']

        # Since both the operations are associative, I have been able to confirm that I can restore back to original values
        # self.data *=self.std
        # self.data +=self.mean
        # logger.info(f'mean # {self.mean}, std # {self.std}')

    def model_init(self):
        self.model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.x_train.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    def model_train(self):
        self.model.fit(self.x_train, self.y_train, epochs=self.epoch, batch_size=16, verbose=0)
        self.test_mse_score, self.test_mae_score = self.model.evaluate(self.x_test, self.y_test)
        logger.info(f' mse score #{self.test_mse_score}, mae score #{self.test_mae_score}')
        outcome = self.model.predict(self.x_test)

        # outcome = outcome * self.ytrain_std.get(key='cpr')
        # outcome = outcome + self.ytrain_mean.get(key='cpr')

        df_reverse = self.x_test * self.test_std
        df_reverse = df_reverse + self.test_mean

        df_out = pd.merge(df_reverse, self.data, left_index=True, right_index=True)
        df_out['outcome'] = outcome
        df_out.to_csv(self.outcome_csv_, float_format='%.2f')

    def model_save(self):
        # keras.models.save_model(self.model_save_path)
        self.model.save(self.model_save_path)

    def model_restore(self):
        self.model = keras.models.load_model(self.model_save_path)
