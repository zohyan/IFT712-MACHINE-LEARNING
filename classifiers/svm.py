import os, sys
from cross_validation.cross_validation import CrossValidation
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics
from sklearn.svm import SVC

class SVMClassifier:

    def __init__(self, kernel='rbf', C=1, gamma=1):
        self.model = SVC(probability=True, kernel=kernel, gamma=gamma, C=C)
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def tunning_model(self, hyperparameters, kfold, metrics):
        cross_validate_model = CrossValidation(self.model, hyperparameters, kfold)
        cross_validate_model.fit_and_predict(self.X_train, self.Y_train, self.X_test, self.Y_test, metrics)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, label="Training", metrics="accuracy"):
        if label:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "accuracy":
            self.metrics.accuracy(self.model, x, y, label)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix(self.model, x, y, label)

        elif metrics == "roc":
            self.metrics.plot_roc(self.model, x, y, label)
