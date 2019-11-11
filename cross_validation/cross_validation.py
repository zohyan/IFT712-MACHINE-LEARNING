from sklearn.model_selection import GridSearchCV

class CrossValidation:

    def __init__(self, model, hyperparameters, cv):
        self.model = model
        self.hyperparameters = hyperparameters
        self.cv = cv
        self.model_after_fitting = None
        self.clf = GridSearchCV(self.model, self.hyperparameters, cv=self.cv)

    def fit_cross_validation(self, x, y):
        self.model_after_fitting = self.clf.fit(x, y)

    def get_best_hyperparams(self):
        return self.model_after_fitting.best_estimator_.get_params()

    def get_clf(self):
        return self.clf