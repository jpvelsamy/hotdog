import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger("ACE")


class AskJunoACE:
    def __init__(self):
        self.k_fold_count = 4
        self.num_epochs = 500
        self.all_mae_histories = []

    def fit_1(self, file_name):
        names = ["reach", "impressions", "results", "amount", "frequency", "clicks", "cpc", "ctr", "cpreach", "cpm",
                 "engagement", "cpr"]
        data = pd.read_csv(file_name, engine='c', dtype='float64', names=names, header=0, skiprows=0)
        #mean = data.mean(axis=0)
        #data -= mean
        #std = data.std(axis=0)
        #data /= std
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
        logger.info(f'Input data #\n{x_test}')
        predictions = model.predict(x_test)
        logger.info(f'Input data #\n{predictions}')
