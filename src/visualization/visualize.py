from pathlib import Path

import numpy as np
from yellowbrick.model_selection import LearningCurve
from yellowbrick.model_selection import ValidationCurve

from src import cv
from src import models_from_dataset
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import info


# def train_test_ratio():
#     pass
#
#     datasets = get_datasets()
#
#     for ds_cnt, filename in enumerate(datasets):
#         dataset = Dataset(filename)
#         info(f"dataset: {dataset.name}")
#
#         for test_size in np.arange(.5, .95, .1):
#             info(f"test_size: {test_size}")
#             dataset.load_dataset(test_size=test_size)
#
#             # iterate over classifiers
#             for model in model_list:
#                 model.fit(dataset)
#
#                 model.score(dataset, train=True)
#                 model.score(dataset, train=False)
#
#                 if not fast_run:
#                     model.tune(dataset)


# https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
def validation_curve():
    project_dir = Path(__file__).resolve().parents[2]
    scoring = 'r2'

    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)
        info(f"dataset: {dataset.name}")

        dataset.load_dataset()
        model_list = models_from_dataset(dataset)

        output_path = Path.joinpath(project_dir, "reports", "figures", dataset.name)
        output_path.mkdir(parents=True, exist_ok=True)

        # iterate over classifiers
        for model in model_list:
            # Load a trained model
            model.load(dataset.name)

            viz = ValidationCurve(model.model, param_name=model.validation_curve_param1,
                                  param_range=model.param1_range, scoring=scoring, cv=cv, n_jobs=14)

            viz.fit(dataset.x, dataset.y)

            filepath = Path.joinpath(output_path, f"{model.title}.png")
            viz.fig.savefig(filepath)
            viz.show()


# https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
def learning_curve():
    project_dir = Path(__file__).resolve().parents[2]
    sizes = np.linspace(0.3, 1.0, 10)
    scoring = 'r2'

    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)
        info(f"dataset: {dataset.name}")

        dataset.load_dataset()
        model_list = models_from_dataset(dataset)

        output_path = Path.joinpath(project_dir, "reports", "figures", dataset.name)
        output_path.mkdir(parents=True, exist_ok=True)

        # iterate over classifiers
        for model in model_list:
            # Load a trained model for dataset
            model.load(dataset.name)

            viz = LearningCurve(model.model, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=14)

            viz.fit(dataset.x, dataset.y)

            filepath = Path.joinpath(output_path, f"{model.title}.png")
            viz.fig.savefig(filepath)
            viz.show()


def visualize_dataset():
    # TODO
    pass


def gen_plots():
    visualize_dataset()
    validation_curve()
    learning_curve()


# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
# https://stackoverflow.com/questions/52349169/plotting-test-valid-and-train-acc-again-epochs-in-sklearn
