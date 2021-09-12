from sklearn.neural_network import MLPClassifier

from models.model import Model


class MLP(Model):
    """
    neural network classifier

    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    """

    def __init__(self, title, fast):
        super().__init__(title, MLPClassifier, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = dict(hidden_layer_sizes=[(10, 10), (25, 25), (50, 50), (100, 100),
                                                                 (10, 10, 10), (25, 15, 10), (50, 25, 10),
                                                                 (100, 50, 10)],
                                             learning_rate_init=[0.1, 0.01, 0.001],
                                             max_iter=[1000, 5000])

        # used for validation curve visualization
        self.validation_curve_param1 = 'learning_rate_init'
        self.param1_range = [0.1, 0.05, 0.01, 0.005, 0.001]
        # self.validation_curve_param2 = 'max_depth'
        # self.param2_range = range(3, 25)

        # default
        max_iter = 5 if fast else 1000
        hidden_layer_sizes = (10, 10) if fast else (100, 100)
        self.model = self.model(alpha=1, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
