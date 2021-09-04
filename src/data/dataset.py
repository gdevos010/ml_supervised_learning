from os import walk
from pathlib import Path

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
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self, fast=False):
        """
        load dataset from file
        """
        project_dir = Path(__file__).resolve().parents[2]
        processed_data = Path.joinpath(project_dir, "data", "processed", self.filename)
        df = feather.read_feather(processed_data)

        # The last column is the labels for both datasets
        x = df.iloc[:, : -1]
        y = df.iloc[:, -1]

        x = StandardScaler().fit_transform(x)

        self.split_dataset(x, y)

    def split_dataset(self, x, y, test_size=0.30, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
