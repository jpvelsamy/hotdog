import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger("ACE")


class TestAskJunoACE:
    def __init__(self):
        self.k_fold_count = 4
        self.num_epochs = 500
        self.all_mae_histories = []

    def fit_1(self, file_name):
        names = ["reach", "impressions", "results", "amount", "frequency", "clicks", "cpc", "ctr", "cpreach", "cpm",
                 "engagement", "cpr"]
        data = pd.read_csv(file_name, engine='c', dtype='float64', names=names, header=0, skiprows=0)
        mean = data.mean(axis=0)
        data -= mean
        std = data.std(axis=0)
        data /= std
        x = data.iloc[:, 0:10]
        y = data.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        model.fit(x_train, y_train, epochs=130, batch_size=16, verbose=0)
        test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
        logger.info(f'mse score #{test_mse_score}, mae score #{test_mae_score}')
        #https://stackoverflow.com/questions/40729162/merging-results-from-model-predict-with-original-pandas-dataframe
        y_hats = model.predict(x_test)
        y_test['preds'] = y_hats

        df_out = pd.merge(data, y_test[['preds']], how='left', left_index=True, right_index=True)

        df_out.to_csv('/home/jpvel/Desktop/outcome.csv', float_format='%.2f')

    def fit_2(self, file_name):
        names = ["reach", "impressions", "results", "amount", "frequency", "clicks", "cpc", "ctr", "cpreach", "cpm",
                 "engagement", "cpr"]
        data = pd.read_csv(file_name, engine='c', dtype='float64', names=names, header=0, skiprows=0)
        mean = data.mean(axis=0)
        data -= mean
        std = data.std(axis=0)
        data /= std
        x = data.iloc[:, 0:10]
        y = data.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        model.fit(x_train, y_train, epochs=130, batch_size=16, verbose=0)
        test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
        logger.info(f'mse score #{test_mse_score}, mae score #{test_mae_score}')
        #https://stackoverflow.com/questions/40729162/merging-results-from-model-predict-with-original-pandas-dataframe
        outcome = model.predict(x_test)
        y_test['preds'] = outcome
        df_out = pd.merge(data, y_test, how='left', left_index=True, right_index=True)
        logger.info(df_out.head(10))
        df_out.to_csv('/home/jpvel/Desktop/outcome2.csv', float_format='%.2f')