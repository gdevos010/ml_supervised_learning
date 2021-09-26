from pathlib import Path

import matplotlib.pyplot as plt

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

        scores = [["Model Type", "F1 Score"]]

        # iterate over models
        for model in model_list:
            # Load a trained model for dataset
            model.load(dataset.name)

            f1, run_time = model.score(dataset, train=False)
            scores.append([model.title, round(f1, 3)])

        save_table(scores, dataset)


def save_table(results, dataset: Dataset):
    fig, ax = plt.subplots()
    table = ax.table(cellText=results, loc='center')

    # modify table
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')

    # display and save table
    project_dir = Path(__file__).resolve().parents[2]
    output_path = Path.joinpath(project_dir, "reports", "figures", dataset.name)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = Path.joinpath(output_path, "test_results.png")
    fig.savefig(filepath, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    init_logger()

    predict()
