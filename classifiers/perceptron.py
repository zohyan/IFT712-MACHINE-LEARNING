import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.linear_model import Perceptron
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics
from cross_validation.cross_validation import CrossValidation

class PerceptronClassifier:

    def __init__(self):
        self.model = Perceptron(tol=0.20, max_iter=1000)
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, x):
        self.train()
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
