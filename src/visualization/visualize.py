from pathlib import Path

from yellowbrick.model_selection import ValidationCurve

from src import model_list
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

    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)
        info(f"dataset: {dataset.name}")

        dataset.load_dataset()

        folderpath = Path.joinpath(project_dir, "reports", "figures", dataset.name)
        folderpath.mkdir(parents=True, exist_ok=True)

        # iterate over classifiers
        for model in model_list:
            # Load a trained model
            model.load(dataset.name)

            viz = ValidationCurve(
                model.model, param_name=model.validation_curve_param1,
                param_range=model.param1_range, cv=5, n_jobs=12)

            viz.fit(dataset.x, dataset.y)

            filepath = Path.joinpath(folderpath, f"{model.title}.png")
            viz.fig.savefig(filepath)
            viz.show()
