import numpy as np
from sklearn.svm import SVC

from models.model import Model
from src.data.dataset import Dataset


class SVM(Model):
    """
    support vector classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    """
    def __init__(self, title: str, dataset: Dataset, kernel: str, fast: bool):
        # switch between classification and regression based on dataset
        if dataset.classification:
            super().__init__(title, SVC, dataset, fast)
        else:
            # TODO
            super().__init__(title, SVC, dataset, fast)

        self.kernel = kernel

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = {'C': np.linspace(0.1, 10, 20),
                                         'gamma': np.logspace(-6, -1, 12)}

        # used for validation curve visualization
        self.validation_curve = {'gamma': np.logspace(-6, -1, 12),
                                 'C': np.linspace(0.1, 10, 20)}

        # set default
        max_iter = 10 if fast else -1
        self.model = self.model(kernel=kernel, max_iter=max_iter)
