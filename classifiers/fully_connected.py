import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics
from sklearn.neural_network import MLPClassifier
from cross_validation.cross_validation import CrossValidation

class FullyConnectedClassifier:

    def __init__(self, hidden_layer_sizes=(10,), activation='relu', alpha=1e-2, learning_rate_init=1e-3, solver='adam'):
        self.fc = MLPClassifier(random_state=None, hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, learning_rate_init=learning_rate_init, solver=solver, max_iter=5000)
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()
        self.cv = None

    def train(self):
        self.fc.fit(self.X_train, self.Y_train)

    def predict(self, x):
        return self.fc.predict(x)

    def tunning_model(self, hyperparameters, kfold, metrics):
        cross_validate_model = CrossValidation(self.fc, hyperparameters, kfold)
        cross_validate_model.fit_and_predict(self.X_train, self.Y_train, self.X_test, self.Y_test, metrics)

    def evaluate(self, label="Training", metrics="accuracy"):
        if label:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "accuracy":
            self.metrics.accuracy(self.fc, x, y, label)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix(self.fc, x, y, label)

        elif metrics == "roc":
            self.metrics.plot_roc(self.fc, x, y, label)