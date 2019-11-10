import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.ensemble import RandomForestClassifier
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics

class RandomForestAlgorithmClassifier:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, x):
        self.train()
        return self.model.predict(x)

    def evaluate(self, training=True, metrics="Accuracy"):
        if training:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "Accuracy":
            self.metrics.accuracy(self.model, x, y, training)

        elif metrics == "Confusion Matrix":
            self.metrics.confusion_matrix(self.model, x, y, training)

model = RandomForestAlgorithmClassifier()
model.train()
model.evaluate(training=False, metrics="Accuracy")