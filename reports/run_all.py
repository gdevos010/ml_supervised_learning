from src.data.make_dataset import make_dataset
from src.models.predict_model import predict
from src.models.train_model import train
from src.utils.logger import init_logger
from src.visualization.visualize import gen_plots

if __name__ == '__main__':
    init_logger()

    make_dataset("EEG")
    make_dataset("mitbih")
    train()
    gen_plots()
    predict()
