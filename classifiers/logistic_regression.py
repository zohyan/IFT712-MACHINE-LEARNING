import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.linear_model import LogisticRegression
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics
from cross_validation.cross_validation import CrossValidation
import numpy as np
import pandas as pd

class LogisticRegressionClassifier:

    def __init__(self, penalty='l1', C=1):
        self.model = LogisticRegression(penalty=penalty, solver="liblinear", C=C)
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()
        self.cross_validated_model = None

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

    def print_combination(self):
        params, mean = [self.cv.get_clf().cv_results_[key] for key in ['params', 'mean_test_score']]
        pty = [p['penalty'] for p in params]
        c = [p['C'] for p in params]
        gridsearch = pd.DataFrame([pd.Series(x) for x in [pty,  np.round(c, decimals=2), np.round(mean*100.0, decimals=2)]]).T
        gridsearch.columns = ['Penalty', 'C', 'Accuracy']
        print(gridsearch)