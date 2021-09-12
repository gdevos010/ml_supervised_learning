from sklearn.tree import DecisionTreeClassifier

from models.model import Model


class DecisionTree(Model):
    """
    Decision Tree classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    """

    def __init__(self, name, fast):
        super().__init__(name, DecisionTreeClassifier, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = dict(max_depth=list(range(3, 25)))

        # used for validation curve visualization
        self.validation_curve_param1 = 'max_depth'
        self.param1_range = range(3, 25)
        # self.validation_curve_param2 = 'max_depth'
        # self.param2_range = range(3, 25)

        # set default
        self.model = self.model(max_depth=5)
