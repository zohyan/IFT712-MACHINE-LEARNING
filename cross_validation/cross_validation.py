import os, sys
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from metrics.metrics import Metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class CrossValidation:

    def __init__(self, model, hyperparameters, kfold):
        self.metrics = Metrics()
        cross_validation = StratifiedKFold(n_splits=kfold, shuffle=True)
        self.clf = GridSearchCV(model, hyperparameters, cv=cross_validation, n_jobs=-1, verbose=1)

    def fit_and_predict(self, x_train, y_train, x_test, y_test, metrics):
        prediction = self.clf.fit(x_train, y_train).best_estimator_.predict(x_test)

        if metrics == "accuracy":
            self.metrics.accuracy_after_validation(prediction, y_test)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix_after_validation(prediction, y_test)

        elif metrics == "roc":
            prob = self.clf.fit(x_train, y_train).best_estimator_.predict_proba(x_test)
            self.metrics.plot_roc_after_validation(prob[:, 1], y_test)

    def get_score(self, x_train, y_train):
        return round(self.clf.score(x_train, y_train) * 100, 2)