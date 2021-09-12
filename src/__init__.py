from sklearn.model_selection import StratifiedShuffleSplit

from models.ada import Boosting
from models.dtree import DecisionTree
from models.knn import KNN
from models.nn import MLP
from models.svm import SVM
from src.utils.logger import info


fast_run = False
info(f"fast_run {fast_run}")

model_list = [
    DecisionTree("Decision Trees", fast_run),
    MLP("Neural Network", fast_run),
    Boosting("k nearest neighbors", fast_run),
    # SVM("Linear Support Vector Machines", "linear", fast_run),
    SVM("RBF Support Vector Machines", "rbf", fast_run),
    SVM("POLY Support Vector Machines", "poly", fast_run),
    KNN("k-Nearest Neighbors", fast_run),
]

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=10)
