from models.boosting import Boosting
from models.dtree import DecisionTree
from models.knn import KNN
from models.nn import MLP
from models.svm import SVM
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import info
from src.utils.logger import init_logger


fast_run = False
info(f"fast_run {fast_run}")

model_list = [
    DecisionTree("Decision Trees", fast_run),
    MLP("Neural Network", fast_run),
    Boosting("k nearest neighbors", fast_run),
    SVM("Linear Support Vector Machines", "linear", fast_run),
    SVM("RBF Support Vector Machines", "rbf", fast_run),
    KNN("k-Nearest Neighbors", fast_run),
]


def train():
    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)
        info(f"dataset: {dataset.name}")
        dataset.load_dataset()

        # iterate over classifiers
        for model in model_list:
            model.fit(dataset)

            model.score(dataset, train=True)
            model.score(dataset, train=False)

            if not fast_run:
                model.tune(dataset)


if __name__ == '__main__':
    init_logger()

    train()
