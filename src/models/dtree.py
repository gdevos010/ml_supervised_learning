import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from src.data.dataset import Dataset
from src.models.model import Model


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
                                             min_weight_fraction_leaf=np.linspace(0.0, .5, 5))

        # used for validation curve visualization
        self.validation_curve = {'max_depth': range(3, 25),
                                 'min_weight_fraction_leaf': np.linspace(0.0, .5, 5)}

        # set default
        self.model = self.model(max_depth=20, criterion='gini')
