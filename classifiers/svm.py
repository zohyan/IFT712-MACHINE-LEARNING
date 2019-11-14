import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics
from sklearn.svm import SVC, LinearSVC

class SVMClassifier:

    def __init__(self):
        self.svm = SVC(probability=True)
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.svm.fit(self.X_train, self.Y_train)

    def predict(self, x):
        self.train()
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