from pathlib import Path

import click
import feather
import pandas as pd
from dotenv import find_dotenv
from dotenv import load_dotenv
from scipy.io import arff

from src.utils.logger import info
from src.utils.logger import init_logger


@click.command()
@click.argument("name")
def main(name):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    info(f"{name}: making final data set from raw data")

    # useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    if name == "EEG":
        input_file = Path.joinpath(project_dir, "data", "raw", "EEG Eye State.arff")
        output_file = Path.joinpath(project_dir, "data", "processed", "EEG.feather")

        data = arff.loadarff(input_file)
        df = pd.DataFrame(data[0])
        df.eyeDetection = df.eyeDetection.astype('int')

        feather.write_dataframe(df, str(output_file))

    elif name == "ptbdb":
        input_file_abnormal = Path.joinpath(project_dir, "data", "raw", "ptbdb_abnormal.csv")
        input_file_normal = Path.joinpath(project_dir, "data", "raw", "ptbdb_normal.csv")
        output_file = Path.joinpath(project_dir, "data", "processed", "ptbdb.feather")

        df_abnormal = pd.read_csv(input_file_abnormal, header=None)
        df_normal = pd.read_csv(input_file_normal, header=None)
        df = pd.concat([df_abnormal, df_normal])

        feather.write_dataframe(df, str(output_file))

    elif name == "mitbih":
        # merge test and train for the purpose of this project
        input_file_train = Path.joinpath(project_dir, "data", "raw", "mitbih_train.csv")
        input_file_test = Path.joinpath(project_dir, "data", "raw", "mitbih_test.csv")
        output_file = Path.joinpath(project_dir, "data", "processed", "MITBIH.feather")

        df_abnormal = pd.read_csv(input_file_train, header=None)
        df_normal = pd.read_csv(input_file_test, header=None)
        df = pd.concat([df_abnormal, df_normal])

        feather.write_dataframe(df, str(output_file))


if __name__ == "__main__":
    init_logger()

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
