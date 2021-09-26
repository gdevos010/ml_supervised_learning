import timeit

import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split

from src.data.dataset import Dataset
from src.models.model import Model
from src.utils.logger import info


class TabNet(Model):
    """
    neural network classifier

    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    """

    def __init__(self, title: str, dataset: Dataset, fast: bool):
        super().__init__(title, TabNetClassifier, dataset, fast)

        # used for RandomizedSearchCV tuning
        self.hyper_param_distribution = {"gamma": [1., 1.5, 2.],
                                         "n_d": [8, 16, 64],
                                         "n_a": [8, 16, 64],
                                         "n_independent": range(1, 6),
                                         "n_shared": range(1, 6),
                                         }

        self.is_pytorch = True

        self.max_epochs = 20000 if not fast else 10

        self.n_width = 64
        self.model = TabNetClassifier(
            n_d=self.n_width, n_a=self.n_width, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"gamma": 0.95,
                              "step_size": 20},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15,
            device_name='cuda'
        )

    def fit(self, dataset: Dataset):
        """
        Fit the default classifier on the dataset
        """
        info(f"\tfitting classifier {self.title}")

        start = timeit.default_timer()

        X_train, X_valid, y_train, y_valid = train_test_split(
            dataset.x_train, dataset.y_train,  test_size=0.2 / 0.6, random_state=0)

        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            max_epochs=self.max_epochs, patience=100,
            batch_size=16384, virtual_batch_size=256,
            eval_metric=['auc', self.scoring_metric]
        )

        stop = timeit.default_timer()
        info(f'\t\ttrain time: {round(stop - start, 3)}s')
