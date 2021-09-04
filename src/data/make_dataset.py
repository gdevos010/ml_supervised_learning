import logging
from pathlib import Path

import click
import feather
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv
from scipy.io import arff


@click.command()
@click.argument("name")
def main(name):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"{name}: making final data set from raw data")

    # useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    if name == "EEG":
        input_file = Path.joinpath(project_dir, "data", "raw", "EEG Eye State.arff")
        output_file = Path.joinpath(project_dir, "data", "processed", "EEG.feather")

        data = arff.loadarff(input_file)
        df = pd.DataFrame(data[0])
        feather.write_dataframe(df, str(output_file))
    elif name == "ECG":
        input_file_abnormal = Path.joinpath(project_dir, "data", "raw", "ptbdb_abnormal.csv")
        input_file_normal = Path.joinpath(project_dir, "data", "raw", "ptbdb_normal.csv")
        output_file = Path.joinpath(project_dir, "data", "processed", "ECG.feather")

        df_abnormal = pd.read_csv(input_file_abnormal)
        df_normal = pd.read_csv(input_file_normal)
        df = pd.concat([df_abnormal, df_normal])
        feather.write_dataframe(df, str(output_file))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
