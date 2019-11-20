import os, sys

from cross_validation.cross_validation import CrossValidation

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics
from sklearn.svm import SVC, LinearSVC

class SVMClassifier:

    def __init__(self, kernel='rbf', C=1, gamma=1):
        self.svm = SVC(probability=True, kernel= kernel , gamma= gamma , C= C )
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.svm.fit(self.X_train, self.Y_train)

    def tunning_model(self, hyperparameters, kfold, metrics):
        self.cv = CrossValidation(self.svm, hyperparameters, kfold)
        self.cv.fit_cross_validation(self.X_train, self.Y_train)
        model_tunned = SVMClassifier(C=self.cv.get_best_hyperparams()['C'], gamma=self.cv.get_best_hyperparams()['gamma'] , kernel='rbf' )
        model_tunned.train()
        print("* After cross validation *")
        model_tunned.evaluate(training=True, metrics=metrics)
        model_tunned.evaluate(training=False, metrics=metrics)

    def predict(self, x):
        return self.svm.predict(x)

    def evaluate(self, training=True, metrics="accuracy"):
        if training:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "accuracy":
            self.metrics.accuracy(self.svm, x, y, training)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix(self.svm, x, y, training)

        elif metrics == "roc":
            self.metrics.plot_roc(self.svm, x, y)