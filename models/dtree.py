from sklearn.tree import DecisionTreeClassifier

from models.model import Model


class DecisionTree(Model):
    """
    Decision Tree classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    """

    def __init__(self, name, fast):
        super().__init__(name, DecisionTreeClassifier, fast)
        self.hyper_param_dist = dict(max_depth=list(range(3, 21)))

        # set default
        self.model = self.model(max_depth=5)
