from sklearn.model_selection import StratifiedShuffleSplit

from models.ada import Boosting
from models.dtree import DecisionTree
from models.knn import KNN
from models.nn import MLP
from models.svm import SVM
from src.data.dataset import Dataset
from src.utils.logger import info
# from models.tabnet import TabNet


fast_run = False
info(f"fast_run {fast_run}")


def models_from_dataset(dataset: Dataset):
    model_list = [
        # TabNet("Tab Net", dataset, fast_run),

        DecisionTree("Decision Trees", dataset, fast_run),
        MLP("Neural Network", dataset, fast_run),
        KNN("Uniform k-Nearest Neighbors", dataset, 'uniform', fast_run),
        KNN("Weighted k-Nearest Neighbors", dataset, 'distance', fast_run),
        Boosting("Ada Boost", dataset, fast_run),
        SVM("RBF Support Vector Machine", dataset, "rbf", fast_run),
        SVM("POLY Support Vector Machine", dataset, "poly", fast_run),

    ]

    return model_list


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=10)
