import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

from models.model import Model
from src.data.dataset import Dataset


class Boosting(Model):
    """
    Ada boosting classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
    """

    def __init__(self, title: str, dataset: Dataset, fast: bool):
        # switch between classification and regression based on dataset
        if dataset.classification:
            super().__init__(title, AdaBoostClassifier, dataset, fast)
        else:
            super().__init__(title, AdaBoostRegressor, dataset, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = {'n_estimators': [10, 50, 100, 200],
                                         'learning_rate': [.25, .5, .75, 1.]}

        # used for validation curve visualization
        self.validation_curve = {'n_estimators': range(25, 500, 25),
                                 'learning_rate': np.linspace(0.1, 1., 10)}

        # set default
        self.model = self.model()
