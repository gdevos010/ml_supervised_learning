from src import models_from_dataset
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import init_logger


def predict():
    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)

        dataset.load_dataset()
        model_list = models_from_dataset(dataset)

        # iterate over models
        for model in model_list:
            # Load a trained model for dataset
            model.load(dataset.name)

            model.score(dataset, train=False)


if __name__ == '__main__':
    init_logger()

    predict()
