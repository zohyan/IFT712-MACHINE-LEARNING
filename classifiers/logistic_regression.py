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
        self.logreg = LogisticRegression(penalty=penalty, C=C)
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()
        self.cv = None

    def train(self):
        self.logreg.fit(self.X_train, self.Y_train)

    def predict(self, x):
        return self.logreg.predict(x)

    def evaluate(self, training=True, metrics="accuracy"):
        if training:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "accuracy":
            self.metrics.accuracy(self.logreg, x, y, training)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix(self.logreg, x, y, training)

        elif metrics == "roc":
            self.metrics.plot_roc(self.logreg, x, y)

    def tunning_model(self, hyperparameters, kfold, metrics):
        self.cv = CrossValidation(self.logreg, hyperparameters, kfold)
        self.cv.fit_cross_validation(self.X_train, self.Y_train)
        model_tunned = LogisticRegressionClassifier(self.cv.get_best_hyperparams()['penalty'], self.cv.get_best_hyperparams()['C'])
        model_tunned.train()
        print("** After cross validation **")
        model_tunned.evaluate(training=True, metrics=metrics)
        model_tunned.evaluate(training=False, metrics=metrics)

    def print_combination(self):
        params, mean = [self.cv.get_clf().cv_results_[key] for key in ['params', 'mean_test_score']]
        pty = [p['penalty'] for p in params]
        c = [p['C'] for p in params]
        gridsearch = pd.DataFrame([pd.Series(x) for x in [pty,  np.round(c,  decimals=2), np.round(mean*100.0, decimals=2)]]).T
        gridsearch.columns = ['Penalty', 'C', 'Accuracy']
        print(gridsearch)

    def cross_validate(self, hyperparameters, kfold):
        cross_validation = CrossValidation(self.logreg, hyperparameters, kfold)
        cross_validation.fit_cross_validation(self.X_train, self.Y_train)
        print('model_after_fitting ', cross_validation.get_clf())