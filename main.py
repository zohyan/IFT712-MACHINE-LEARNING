import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
import numpy as np
from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.perceptron import PerceptronClassifier
from classifiers.random_forest import RandomForestAlgorithmClassifier
from classifiers.svm import SVMClassifier
from classifiers.fully_connected import FullyConnectedClassifier
from classifiers.adaboost import AdaBoostAlgorithmClassifier
from classifiers.decision_tree import DecisionTreeAlgorithmClassifier
from classifiers.bagging import BaggingAlgorithmClassifier

def main():
    if sys.argv[2] in ['accuracy', 'confusion_matrix', 'roc']:

        if sys.argv[1] == 'logistic_regression':
            model = LogisticRegressionClassifier()
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5
                hyperparameters = {
                    'penalty': ['l1', 'l2'],
                    'C': np.logspace(0, 4, 10)
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'perceptron':
            model = PerceptronClassifier()
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5
                hyperparameters = {
                    'penalty': ['l1', 'l2'],
                    'alpha': [1e-4, 0.004, 0.005, 0.008],
                    'tol': [0.20, 0.19, 0.15, 0.25]
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'random_forest':
            model = RandomForestAlgorithmClassifier()
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5

                n_estimators = [int(x) for x in np.linspace(start=200, stop=500 , num=10)]
                max_features = ['auto', 'sqrt']
                max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
                max_depth.append(None)
                min_samples_split = [2, 5, 10]
                min_samples_leaf = [1, 2, 4]
                bootstrap = [True, False]

                hyperparameters = {
                    'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'svm':
            model = SVMClassifier()

            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5
                hyperparameters = {
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel':  ['rbf']
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'fully_connected':
            model = FullyConnectedClassifier()

            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5
                hyperparameters = {
                    'hidden_layer_sizes': [(5, ), (5, 5)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [1e-5, 3e-4, 7e-2],
                    'learning_rate_init':  [1e-2, 1e-3, 1e-4]
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'adaboost':
            model = AdaBoostAlgorithmClassifier()
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5
                hyperparameters = {
                    'base_estimator': [DecisionTreeAlgorithmClassifier().model],
                    'n_estimators': [50, 55, 60, 65, 70],
                    'algorithm': ['SAMME.R', 'SAMME']
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'decision_tree':
            model = DecisionTreeAlgorithmClassifier()
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5
                hyperparameters = {
                    'criterion': ['entropy', 'gini'],
                    'max_depth': [2, 4]
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'bagging':
            model = BaggingAlgorithmClassifier()
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            if sys.argv[3] == '1':
                kfold = 5
                hyperparameters = {
                    'weights': [[int(x) for x in list("{0:0b}".format(i).zfill(4))] for i in range(1, 2 ** 4)]
                }
                model.tunning_model(hyperparameters, kfold, sys.argv[2])

if __name__ == "__main__":
    main()
