from sklearn.ensemble import AdaBoostClassifier

from models.model import Model


class Boosting(Model):
    """
    Ada boosting classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
    """

    def __init__(self, title, fast):
        super().__init__(title, AdaBoostClassifier, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = {'n_estimators': [10, 50, 100, 200],
                                         'learning_rate': [.5, 1, 1.5, 2]}

        # used for validation curve visualization
        self.validation_curve_param1 = 'n_estimators'
        self.param1_range = range(10, 500, 10)
        # self.validation_curve_param2 = 'n_estimators'
        # self.param2_range = range(10, 200, 10)

        # set default
        self.model = self.model()
