import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils.data_preprocessing import DataPreprocessing
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from metrics.metrics import Metrics
from cross_validation.cross_validation import CrossValidation

class BaggingAlgorithmClassifier:

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.svm = SVC(C=5, gamma="scale", probability=True)
        self.fc_tanh = MLPClassifier(hidden_layer_sizes=(10, 10, ), activation='tanh', max_iter=5000)
        self.fc_relu = MLPClassifier(hidden_layer_sizes=(10, 10, ), activation='relu', max_iter=5000)
        self.clf_array = [self.rf, self.svm, self.fc_tanh, self.fc_relu]
        self.model = None
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.model = VotingClassifier(estimators=[
            ('Random Forests', self.rf),
            ('SVM', self.svm),
            ('FC_tanh', self.fc_tanh),
            ('FC_relu', self.fc_relu)
        ], voting='soft')

        self.model.fit(self.X_train, self.Y_train)

    def predict(self, x):
        return self.model.predict(x)

    def tunning_model(self, hyperparameters, kfold, metrics):
        cross_validate_model = CrossValidation(self.model, hyperparameters, kfold)
        cross_validate_model.fit_and_predict(self.X_train, self.Y_train, self.X_test, self.Y_test, metrics)

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
