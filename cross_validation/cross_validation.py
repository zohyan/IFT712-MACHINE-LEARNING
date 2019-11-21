import os, sys
from sklearn.model_selection import GridSearchCV
from metrics.metrics import Metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

class CrossValidation:

    def __init__(self, model, hyperparameters, cv):
        self.metrics = Metrics()
        self.clf = GridSearchCV(model, hyperparameters, cv=cv)

    def fit_and_predict(self, x_train, y_train, x_test, y_test, metrics):
        prediction = self.clf.fit(x_train, y_train).best_estimator_.predict(x_test)

        if metrics == "accuracy":
            self.metrics.accuracy_after_validation(prediction, y_test)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix_after_validation(prediction, y_test)

        elif metrics == "roc":
            prob = self.clf.fit(x_train, y_train).best_estimator_.predict_proba(x_test)
            self.metrics.plot_roc_after_validation(prob[:, 1], y_test)

    def get_clf(self):
        return self.clf