import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing

logger = logging.getLogger("ACE")


class BostonRegression:


    def __init__(self):

        self.num_epochs = 5
        self.all_mae_histories = []
        self.model = None
        logger.info("Loading regression dataset")
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = boston_housing.load_data()

        self.mean = self.train_data.mean(axis=0)
        self.train_data -= self.mean
        self.std = self.train_data.std(axis=0)
        self.train_data /= self.std
        self.test_data -= self.mean
        self.test_data /= self.std

    def showbostonmeta(self):
        logger.info(f"Training data shape # {self.train_data.shape}")
        logger.info(f"Test data shape # {self.test_data.shape}")
        logger.info(f"Train targets # {self.train_targets}")
        boston_dataset = load_boston()
        boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
        boston['MEDV'] = boston_dataset.target
        head = boston.head()
        logger.info(f"\n{head}")

    def showmetadata(self):
        housing = fetch_california_housing()
        keys = housing.keys()
        logger.info(f"Data set # {keys}")

    def showhead(self):
        housing = fetch_california_housing()
        cali = pd.DataFrame(housing.data, columns=housing.feature_names)
        head = cali.head()
        logger.info(f"\n{head}")

    def build_model(self):
        self.model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(self.train_data.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    def kfold_validation(self):

        k = 4
        num_val_samples = len(self.train_data) // k
        all_scores = []
        for i in range(k):
            logger.info(f"Processing fold #{i}")
            val_data = self.train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.train_targets[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [self.train_data[:i * num_val_samples],
                 self.train_data[(i + 1) * num_val_samples:]],
                axis=0)
            partial_train_targets = np.concatenate(
                [self.train_targets[:i * num_val_samples],
                 self.train_targets[(i + 1) * num_val_samples:]],
                axis=0)
            model = keras.Sequential([
                layers.Dense(64, activation="relu", input_shape=(self.train_data.shape[1],)),
                layers.Dense(64, activation="relu"),
                layers.Dense(1)
            ])
            model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
            history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                                epochs=self.num_epochs, batch_size=1, verbose=0)
            mae_history = history.history["val_mae"]
            self.all_mae_histories.append(mae_history)
            val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
            all_scores.append(val_mae)
            self.all_mae_histories.append(mae_history)

    def plot_kfold(self):
        average_mae_history = [np.mean([x[i] for x in self.all_mae_histories]) for i in range(self.num_epochs)]
        plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
        plt.xlabel('Epochs')
        plt.ylabel('Validation MAE')
        plt.show()

    def train_model(self):
        self.model.fit(self.train_data, self.train_targets, epochs=130, batch_size=16, verbose=0)
        self.test_mse_score, self.test_mae_score = self.model.evaluate(self.test_data, self.test_targets)

    def test(self):
        logger.info(f'Input data #\n{self.test_data}')
        predictions = self.model.predict(self.test_data)
        logger.info(f'Input data #\n{predictions}')
