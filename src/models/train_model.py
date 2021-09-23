from pathlib import Path

import matplotlib.pyplot as plt

from src import fast_run
from src import models_from_dataset
from src.data.dataset import Dataset
from src.data.dataset import get_datasets
from src.utils.logger import init_logger


def train():
    """
    loop over each processed dataset. fit, tune and save the model
    """
    datasets = get_datasets()

    for ds_cnt, filename in enumerate(datasets):
        dataset = Dataset(filename)

        dataset.load_dataset()
        model_list = models_from_dataset(dataset)

        times = [["Model Type", "Train Time", "Predict Time"]]
        # scores = [["Model Type", "F1 Train", "F1 Test"]]

        # iterate over models
        for model in model_list:
            train_time = model.fit(dataset)

            f1_train, run_time_train = model.score(dataset, train=True)
            f1_test, run_time_test = model.score(dataset, train=False)

            if not fast_run:
                model.tune(dataset)

                f1_train, run_time_train = model.score(dataset, train=True)
                f1_test, run_time_test = model.score(dataset, train=False)

            times.append([model.title, train_time, round(run_time_test, 3)])
        save_table(times, dataset)


def save_table(results, dataset: Dataset):
    fig, ax = plt.subplots()
    table = ax.table(cellText=results, loc='center')

    # modify table
    table.set_fontsize(14)
    table.scale(1, 4)
    ax.axis('off')

    # display and save table
    project_dir = Path(__file__).resolve().parents[2]
    output_path = Path.joinpath(project_dir, "reports", "figures", dataset.name)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = Path.joinpath(output_path, "run_times.png")
    fig.savefig(filepath)
    plt.show()


if __name__ == '__main__':
    init_logger()

    train()
