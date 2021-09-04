from models.models import classifiers
from models.models import names
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import info
from src.utils.logger import initLogger


def train():

    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)
        info(f"dataset: {dataset.name}")
        dataset.load_dataset()

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            info(f"fitting classifier {name}")
            clf.fit(dataset.x_train, dataset.y_train)

            # get the mean accuracy on the train data
            score = clf.score(dataset.x_train, dataset.y_train)
            info(f"    classifier score {round(score,3)} on training set")
            # get the mean accuracy on the test data
            score = clf.score(dataset.x_test, dataset.y_test)
            info(f"    classifier score {round(score,3)} on test set")


if __name__ == '__main__':
    initLogger()

    train()
