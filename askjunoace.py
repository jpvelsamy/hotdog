import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from hotdogconfig import Configuration

logger = logging.getLogger("ACE")


class AskJunoACE:
    def __init__(self, config_object:Configuration, sigma_folder):
        self.config_object = config_object
        self.sigma_folder = sigma_folder
        self.cpl_train_file = self.sigma_folder+self.config_object.get_path_separator()+self.config_object.get_input_file_name()
        myfeatures = self.config_object.get_cpl_features()
        self.features = myfeatures.split(",")
        self.ratio = float(self.config_object.get_cpl_ratio())
        self.outcome_file = self.config_object.get_cpl_outcome()
        self.model_save_path = self.config_object.get_model_save_path()+'/ace_cpl.h5'

    def load_data(self):
        self.data = pd.read_csv(self.cpl_train_file, engine='c', dtype='float64', names=self.features
                                , header=0, skiprows=0)

    def normalize_data(self):
        self.mean = self.data.mean(axis=0)
        self.data -= self.mean
        self.std = self.data.std(axis=0)
        self.data /= self.std
        logger.info(f'mean # {self.mean}, std # {self.std}')

    def test_train_split(self):
        x = self.data.iloc[:, 0:11]
        y = self.data.iloc[:, -1]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.ratio)

    def model_init(self):
        self.model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.x_train.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    def model_train(self):
        self.model.fit(self.x_train, self.y_train, epochs=130, batch_size=16, verbose=0)
        test_mse_score, test_mae_score = self.model.evaluate(self.x_test, self.y_test)
        logger.info(f'mse score #{test_mse_score}, mae score #{test_mae_score}')
        # https://stackoverflow.com/questions/40729162/merging-results-from-model-predict-with-original-pandas-dataframe
        outcome = self.model.predict(self.x_test)
        outcome = outcome * self.std.get(key='cpr')
        outcome = outcome + self.mean.get(key='cpr')

        df_reverse = self.x_test * self.std
        df_reverse = df_reverse + self.mean

        df_final = self.data * self.std
        df_final = df_final + self.mean

        df_out = pd.merge(df_reverse, df_final, left_index=True, right_index=True)
        df_out['outcome'] = outcome
        logger.info(df_out.head(10))
        df_out.to_csv(self.outcome_file+self.config_object.get_path_separator()+self.config_object.get_input_file_name()+'_outcome.csv', float_format='%.2f')

    def model_save(self):
        # keras.models.save_model(self.model_save_path)
        self.model.save(self.model_save_path)

    def model_restore(self):
        self.model = keras.models.load_model(self.model_save_path)
