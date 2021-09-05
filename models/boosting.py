from sklearn.ensemble import AdaBoostClassifier

from models.model import Model


class Boosting(Model):
    """
    Ada boosting classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
    """
    def __init__(self, title, fast):
        super().__init__(title, AdaBoostClassifier, fast)
        self.hyper_param_dist = {'n_estimators': [10, 50, 100, 200],
                                 'learning_rate': [.5, 1, 1.5, 2]}

        # set default
        self.model = self.model()
