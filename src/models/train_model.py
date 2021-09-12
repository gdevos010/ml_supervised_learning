from src import fast_run
from src import model_list
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import info
from src.utils.logger import init_logger
from src.visualization.visualize import validation_curve


def train():
    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)
        info(f"dataset: {dataset.name}")
        dataset.load_dataset()

        # iterate over classifiers
        for model in model_list:
            model.fit(dataset)

            model.score(dataset, train=True)
            model.score(dataset, train=False)

            if not fast_run:
                model.tune(dataset)


def gen_plots():
    validation_curve()
    pass


if __name__ == '__main__':
    init_logger()

    # train()
    gen_plots()
