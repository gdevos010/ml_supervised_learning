import pickle
import timeit
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV

from src.data.dataset import Dataset
from src.utils.logger import info


class Model:

    def __init__(self, title: str, model, dataset: Dataset, fast: bool):
        self.title = title
        self.model = model
        self.hyper_param_distribution = None
        self.validation_curve = None
        self.fast = fast

        self.feature_count = dataset.feature_count
        self.class_num = dataset.class_num

        self.is_pytorch = False

        self.n_iter_search = 20
        if self.fast:
            self.n_iter_search = 3

    def __str__(self):
        return self.title

    def __lt__(self, other):
        return self.title < other.title

    def fit(self, dataset: Dataset):
        """
        Fit the default classifier on the dataset
        """
        info(f"\tfitting classifier {self.title}")

        start = timeit.default_timer()

        self.model.fit(dataset.x_train, dataset.y_train)

        stop = timeit.default_timer()
        info(f'\t\ttrain time: {round(stop - start, 3)}s')

    def score(self, dataset: Dataset, train):
        """
        calculate the accuracy on the dataset
        """
        start = timeit.default_timer()

        # get the mean accuracy on the dataset
        if train:
            x = dataset.x_train
            y = dataset.y_train
            mode = 'train'
        else:
            x = dataset.x_test
            y = dataset.y_test
            mode = 'test'

        score = self.model.score(x, y)
        stop = timeit.default_timer()

        info(f"\t\tclassifier score {round(score, 3)} on {mode} set ({round(stop - start, 3)}s)")

    def save(self, dataset_name: str):
        """
        save model under models/zoo
        """
        project_dir = Path(__file__).resolve().parents[1]
        filepath = Path.joinpath(project_dir, "models", "zoo", dataset_name)
        filepath.mkdir(parents=True, exist_ok=True)
        filepath = Path.joinpath(filepath, self.title + ".pkl")

        pickle.dump(self.model, open(filepath, 'wb'))

    def load(self, dataset_name: str):
        """
        load model from models/zoo
        """
        info(f"\tloading: {self.title}")
        project_dir = Path(__file__).resolve().parents[1]
        filepath = Path.joinpath(project_dir, "models", "zoo", dataset_name,  self.title + ".pkl")
        self.model = pickle.load(open(filepath, 'rb'))

    def tune(self, dataset: Dataset, verbose=0):
        """
        use random search to find best model
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        """
        info(f'\ttuning: {self.title}')
        start = timeit.default_timer()

        clf = RandomizedSearchCV(self.model, self.hyper_param_distribution, random_state=0,
                                 n_iter=self.n_iter_search, n_jobs=14, verbose=verbose)
        search = clf.fit(dataset.x_train, dataset.y_train)

        stop = timeit.default_timer()
        info(f'\t\tsearch time: {round(stop - start, 3)}s')

        # info(f"\t\tbest score {round(search.best_score_, 3)}")
        info(f"\t\t{search.best_params_}")

        # save the best model
        self.model = search.best_estimator_
        self.save(dataset.name)
