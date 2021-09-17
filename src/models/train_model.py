from src import fast_run
from src import models_from_dataset
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import init_logger


def train():
    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)

        dataset.load_dataset()
        model_list = models_from_dataset(dataset)

        # iterate over models
        for model in model_list:
            model.fit(dataset)

            model.score(dataset, train=True)
            model.score(dataset, train=False)

            if not fast_run:
                model.tune(dataset)


if __name__ == '__main__':
    init_logger()

    train()
