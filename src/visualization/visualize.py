import multiprocessing as mp
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from yellowbrick.model_selection import LearningCurve
from yellowbrick.model_selection import ValidationCurve

from src import cv
from src import models_from_dataset
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import info
from src.utils.logger import init_logger


def loss_curve():
    # TODO
    pass


# https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
def validation_curve():
    project_dir = Path(__file__).resolve().parents[2]

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
                # skip if visualization already exists
                filepath = Path.joinpath(output_path, f"{model.title} {param_name}.png")
                if filepath.is_file():
                    info(f"viz exists. skipping: {filepath}")
                    continue

                # Load a trained model
                model.load(dataset.name)

                chart_title = f"Validation Curve for {model.title} ({dataset.name})"
                viz = ValidationCurve(model.model, param_name=param_name,
                                      param_range=param_range, scoring=model.scoring_metric, cv=cv,
                                      title=chart_title, n_jobs=mp.cpu_count() - 1)

                viz.fit(dataset.x, dataset.y)
                best_test_scores = np.argmax(viz.test_scores_mean_)
                info(f"best {param_name} is {param_range[best_test_scores]} with a score of"
                     f" {round(viz.test_scores_mean_[best_test_scores],3)}")

                viz.show(outpath=filepath)
                viz.show(clear_figure=True)
                # plt.gcf().clear()


# https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
def learning_curve():
    project_dir = Path(__file__).resolve().parents[2]
    sizes = np.arange(0.3, 1., .2)

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
            # skip if visualization already exists
            filepath = Path.joinpath(output_path, f"{model.title} learning curve.png")
            if filepath.is_file():
                info(f"viz exists. skipping: {filepath}")
                continue

            # Load a trained model for dataset
            model.load(dataset.name)

            chart_title = f"Learning Curve for {model.title} ({dataset.name})"
            viz = LearningCurve(model.model, cv=cv, scoring=model.scoring_metric, train_sizes=sizes,
                                title=chart_title, n_jobs=mp.cpu_count() - 1)

            viz.fit(dataset.x, dataset.y)

            best_test_scores = np.argmax(viz.test_scores_mean_)
            info(f"best split is {round(sizes[best_test_scores],3)} with a score of "
                 f"{round(viz.test_scores_mean_[best_test_scores],3)}")

            viz.show(outpath=filepath)
            viz.show(clear_filename=True)


def visualize_dataset():
    # TODO generalize
    # class split
    project_dir = Path(__file__).resolve().parents[2]

    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset: Dataset = Dataset(filename)
        info(f"dataset: {dataset.name}")

        output_path = Path.joinpath(project_dir, "reports", "figures", dataset.name)
        output_path.mkdir(parents=True, exist_ok=True)

        # skip if visualization already exists
        filepath = Path.joinpath(output_path, "class distribution.png")
        if filepath.is_file():
            info(f"viz exists. skipping: {filepath}")
            continue

        dataset.load_dataset()

        # plot from https://www.kaggle.com/gregoiredc/arrhythmia-on-ecg-classification-using-cnn/notebook
        plt.figure(figsize=(10, 10))
        my_circle = plt.Circle((0, 0), 0.7, color='white')

        if dataset.name == "EEG":
            # Classes:
            # '1' indicates the eye-closed
            # '0' the eye-open state
            unique, counts = np.unique(dataset.y, return_counts=True)

            plt.pie(counts, labels=['eye-open', 'eye-closed'], colors=['red', 'green'],
                    autopct='%1.1f%%')

        if dataset.name == "mitbih":
            # Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
            # -N : Non-ecotic beats (normal beat)
            # -S : Supraventricular ectopic beats
            # -V : Ventricular ectopic beats
            # -F : Fusion Beats
            # -Q : Unknown Beats
            unique, counts = np.unique(dataset.y, return_counts=True)

            plt.pie(counts, labels=['N', 'S', 'V', 'F', 'Q'], colors=['red', 'green', 'blue', 'skyblue', 'orange'],
                    autopct='%1.1f%%')

        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.title(f'Class Distribution for {dataset.name}')
        plt.savefig(filepath, bbox_inches='tight')
        plt.show()


def gen_plots():
    visualize_dataset()
    validation_curve()
    learning_curve()


if __name__ == '__main__':
    init_logger()

    gen_plots()

# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
# https://stackoverflow.com/questions/52349169/plotting-test-valid-and-train-acc-again-epochs-in-sklearn
