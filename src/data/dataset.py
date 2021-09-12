from os import walk
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_datasets():
    """
    get all datasets in data/processed folder
    """
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_folder = Path.joinpath(project_dir, "data", "processed")
    filenames = next(walk(processed_data_folder), (None, None, []))[2]
    filenames = list(filter(lambda x: 'feather' in x, filenames))
    return filenames


class Dataset:
    def __init__(self, filename: str):
        self.name = filename[:-8]
        self.filename = filename
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def __str__(self):
        return self.name

    def load_dataset(self, fast=False, test_size=0.3):
        """
        load dataset from file
        """
        project_dir = Path(__file__).resolve().parents[2]
        processed_data = Path.joinpath(project_dir, "data", "processed", self.filename)
        df = feather.read_feather(processed_data)

        # The last column is the labels for both datasets
        x = df.iloc[:, : -1]
        y = df.iloc[:, -1]

        self.split_dataset(x, y, test_size)

        # preprocess dataset.
        # Fit standard scaler on training set and transform test set
        standard_scaler = StandardScaler()
        self.x_train = standard_scaler.fit_transform(self.x_train)
        self.x_test = standard_scaler.transform(self.x_test)

        # some plots require just an x and y
        self.x = np.vstack((self.x_train, self.x_test))
        self.y = np.concatenate((self.y_train, self.y_test))

    def split_dataset(self, x, y, test_size=0.30, random_state=42):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train_test_split#sklearn.model_selection.train_test_split
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,
                                                                                y,
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        self.y_train = self.y_train.to_numpy()
        self.y_test = self.y_test.to_numpy()
