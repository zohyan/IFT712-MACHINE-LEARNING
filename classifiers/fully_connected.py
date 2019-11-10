import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics
from sklearn.neural_network import MLPClassifier

class FullyConnectedClassifier:

    def __init__(self):
        self.fc = MLPClassifier()
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.fc.fit(self.X_train, self.Y_train)

    def predict(self, x):
        self.train()
        return self.fc.predict(x)

    def evaluate(self, training=True, metrics="Accuracy"):
        if training:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "Accuracy":
            self.metrics.accuracy(self.fc, x, y, training)

        elif metrics == "Log Loss":
            self.metrics.log_loss(self.fc, x, y, training)

        elif metrics == "Confusion Matrix":
            self.metrics.confusion_matrix(self.fc, x, y, training)

FC = FullyConnectedClassifier()
FC.train()
FC.evaluate(training=False, metrics="Confusion Matrix")