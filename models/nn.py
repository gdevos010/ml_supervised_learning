import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from models.model import Model
from src.data.dataset import Dataset


class MLP(Model):
    """
    neural network classifier

    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    """

    def __init__(self, title: str, dataset: Dataset, fast: bool):
        # switch between classification and regression based on dataset
        if dataset.classification:
            super().__init__(title, MLPClassifier, dataset, fast)
        else:
            super().__init__(title, MLPRegressor, dataset, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = dict(hidden_layer_sizes=[(10, 10), (25, 25), (50, 50), (100, 100),
                                                                 (10, 10, 10), (25, 15, 10), (50, 25, 10),
                                                                 (100, 50, 10)],
                                             learning_rate_init=[0.1, 0.01, 0.001],
                                             max_iter=[10000],
                                             momentum=[.25, .5, .75, .9])

        # used for validation curve visualization
        self.validation_curve = {"learning_rate_init": [0.001, 0.005, 0.01, 0.05, 0.1],
                                 "momentum": np.linspace(0.1, 1., 10),
                                 "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]}

        # default
        max_iter = 5 if fast else 10000
        hidden_layer_sizes = (10, 10) if fast else (100, 100)
        self.model = self.model(max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes,
                                early_stopping=True, n_iter_no_change=15)
