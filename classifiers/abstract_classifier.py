import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from metrics.metrics import Metrics
from data_utils.data_preprocessing import DataPreprocessing
from cross_validation.cross_validation import CrossValidation


class AbstractClassifier:

    """
    Parent class of all project classifiers.

    Attributes:
        model : An object that defines the classifier model to implement.
        metrics : An object that defines the different metrics that can be used to evaluate a model.
        X_train : The features of the training data
        Y_train : The targets of training data (the ground truth label)
        X_test :  The features of the testing data
        Y_test : The targets of training data (the ground truth label)
    """

    def __init__(self, model):
        self.model = model
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().advanced_preprocessing_data()
        # self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, label="Training", metrics="accuracy"):
        if label == 'Training':
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "accuracy":
            self.metrics.accuracy(self.model, x, y, label)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix(self.model, x, y, label)

        elif metrics == "roc":
            self.metrics.plot_roc(self.model, x, y, label)

    def tunning_model(self, hyperparameters, kfold, metrics):
        cross_validate_model = CrossValidation(self.model, hyperparameters, kfold)
        cross_validate_model.fit_and_predict(self.X_train, self.Y_train, self.X_test, self.Y_test, metrics)
        return cross_validate_model.get_score(self.X_train, self.Y_train)
