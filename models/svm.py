import numpy as np
from sklearn.svm import SVC

from models.model import Model


class SVM(Model):
    """
    support vector classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    """
    def __init__(self, title,  kernel, fast):
        super().__init__(title, SVC, fast)
        self.kernel = kernel

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = {'C': [.5, 1., 2, 5, 10],
                                         'gamma': np.logspace(-6, -1, 12),
                                         'coef0': [0, .1, .5, 1]}

        # used for validation curve visualization
        self.validation_curve_param1 = 'gamma'
        self.param1_range = np.logspace(-6, -1, 12)
        # self.validation_curve_param2 = 'max_depth'
        # self.param2_range = range(3, 25)

        # set default
        max_iter = 10 if fast else -1
        self.model = self.model(kernel=kernel, max_iter=max_iter)
