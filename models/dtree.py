import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from models.model import Model
from src.data.dataset import Dataset


class DecisionTree(Model):
    """
    Decision Tree classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    """

    def __init__(self, title: str, dataset: Dataset, fast: bool):
        # switch between classification and regression based on dataset
        if dataset.classification:
            super().__init__(title, DecisionTreeClassifier, dataset, fast)
        else:
            super().__init__(title, DecisionTreeRegressor, dataset, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = dict(max_depth=np.arange(3, 25),
                                             min_samples_leaf=np.logspace(-4, -1, 10).tolist() + [1])

        # used for validation curve visualization
        self.validation_curve_param1 = 'max_depth'
        self.param1_range = range(3, 25)
        # self.validation_curve_param2 = 'max_depth'
        # self.param2_range = range(3, 25)

        # set default
        self.model = self.model(max_depth=5, criterion='gini')
