from sklearn.svm import SVC

from models.model import Model


class SVM(Model):
    """
    support vector classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    """
    def __init__(self, title,  kernel, fast):
        super().__init__(title, SVC, fast)
        self.kernel = kernel

        self.hyper_param_dist = {'C': [1, 10]}

        # set default
        max_iter = 10 if fast else -1
        self.model = self.model(kernel=kernel, max_iter=max_iter)
