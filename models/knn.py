from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from models.model import Model
from src.data.dataset import Dataset


class KNN(Model):
    """
    K Neighbors Classifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    """

    def __init__(self, title: str, dataset: Dataset, fast: bool):
        # switch between classification and regression based on dataset
        if dataset.classification:
            super().__init__(title, KNeighborsClassifier, dataset, fast)
        else:
            super().__init__(title, KNeighborsRegressor, dataset, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = dict(n_neighbors=list(range(1, 15, 2)),
                                             weights=["uniform", "distance"])

        # used for validation curve visualization
        self.validation_curve = {'n_neighbors': range(1, 15, 2),
                                 'weights': ["uniform", "distance"]}

        # set default
        self.model = self.model(3, n_jobs=-1)
