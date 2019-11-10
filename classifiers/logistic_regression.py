import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.linear_model import LogisticRegression
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics

class LogisticRegressionClassifier:

    def __init__(self):
        self.logreg = LogisticRegression()
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.logreg.fit(self.X_train, self.Y_train)

    def predict(self, x):
        self.train()
        return self.logreg.predict(x)

    def evaluate(self, training=True, metrics="Accuracy"):
        if training:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "Accuracy":
            self.metrics.accuracy(self.logreg, x, y, training)

        elif metrics == "Log Loss":
            self.metrics.log_loss(self.logreg, x, y, training)

        elif metrics == "Confusion Matrix":
            self.metrics.confusion_matrix(self.logreg, x, y, training)

LR = LogisticRegressionClassifier()
LR.train()
LR.evaluate(training=False, metrics="Confusion Matrix")