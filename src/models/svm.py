import numpy as np
from sklearn.svm import SVC

from src.data.dataset import Dataset
from src.models.model import Model


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
        self.max_iter = 10000

        gamma_range = np.linspace(0.01, 10, 5)
        c_range = np.linspace(0.01, 10, 5)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = {'C': np.linspace(0.1, 15, 5),
                                         'gamma': np.linspace(0.1, 100, 5),
                                         'max_iter': [self.max_iter]}

        # used for validation curve visualization
        self.validation_curve = {'gamma': gamma_range,
                                 'C': c_range}  # decreasing C corresponds to more regularization

        # set default
        max_iter = 10 if fast else self.max_iter
        self.model = self.model(kernel=kernel, max_iter=max_iter, cache_size=1500, gamma=.1, C=3.825)
