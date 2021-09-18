from pathlib import Path

import numpy as np
from yellowbrick.model_selection import LearningCurve
from yellowbrick.model_selection import ValidationCurve

from src import cv
from src import models_from_dataset
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import info


def loss_curve():
    # TODO
    pass


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
            for param_name, param_range in model.validation_curve.items():
                # Load a trained model
                model.load(dataset.name)

                viz = ValidationCurve(model.model, param_name=param_name,
                                      param_range=param_range, scoring=scoring, cv=cv, n_jobs=14)

                viz.fit(dataset.x, dataset.y)

                filepath = Path.joinpath(output_path, f"{model.title} {param_name}.png")
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
    # class split
    pass


def gen_plots():
    visualize_dataset()
    validation_curve()
    learning_curve()


if __name__ == '__main__':
    gen_plots()

# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
# https://stackoverflow.com/questions/52349169/plotting-test-valid-and-train-acc-again-epochs-in-sklearn
