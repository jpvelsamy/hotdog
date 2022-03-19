import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger("ACE")


class AskJunoACE:
    def __init__(self, config_object):
        self.config_object = config_object
        self.cpltrainfile = self.config_object.get_cpl_train_file()
        self.names = ["reach", "impressions", "results", "amount", "frequency", "clicks", "cpc", "ctr", "cpreach",
                      "cpm",
                      "engagement", "cpr"]
        self.features = self.config_object.get_cpl_features()
        self.ratio = self.config_object.get_cpl_ratio()
        self.outcome_file = self.config_object.get_outcome_file()

    def load_data(self):
        self.data = pd.read_csv(self.cpltrainfile, engine='c', dtype='float64', names=self.names, header=0, skiprows=0)

    def normalize_data(self):
        mean = self.data.mean(axis=0)
        self.data -= mean
        std = self.data.std(axis=0)
        self.data /= std

    def test_train_split(self):
        x = self.data.iloc[:, 0:10]
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
        y_hats = self.model.predict(self.x_test)
        self.y_test['preds'] = y_hats
        df_out = pd.merge(self.data, self.y_test[['preds']], how='left', left_index=True, right_index=True)
        df_out.to_csv(self.outcome_file, float_format='%.2f')
