from sklearn.neighbors import KNeighborsClassifier

from models.model import Model


class KNN(Model):
    """
    K Neighbors Classifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    """

    def __init__(self, title, fast):
        super().__init__(title, KNeighborsClassifier, fast)
        self.hyper_param_dist = dict(n_neighbors=list(range(1, 15, 2)))

        # set default
        self.model = self.model(3, n_jobs=-1)
