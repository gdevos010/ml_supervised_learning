from sklearn.model_selection import StratifiedShuffleSplit

from models.ada import Boosting
from models.dtree import DecisionTree
from models.knn import KNN
from models.nn import MLP
from models.svm import SVM
from src.data.dataset import Dataset
from src.utils.logger import info


fast_run = False
info(f"fast_run {fast_run}")


def models_from_dataset(dataset: Dataset):
    model_list = [
        DecisionTree("Decision Trees", dataset, fast_run),
        MLP("Neural Network", dataset, fast_run),
        Boosting("k nearest neighbors", dataset, fast_run),
        SVM("RBF Support Vector Machines", dataset, "rbf", fast_run),
        SVM("POLY Support Vector Machines", dataset, "poly", fast_run),
        KNN("k-Nearest Neighbors", dataset, fast_run),
    ]

    return model_list


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=10)
