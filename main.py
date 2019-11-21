import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.perceptron import PerceptronClassifier
from classifiers.random_forest import RandomForestAlgorithmClassifier
from classifiers.svm import SVMClassifier
from classifiers.fully_connected import FullyConnectedClassifier
from classifiers.adaboost import AdaBoostAlgorithmClassifier
from classifiers.decision_tree import DecisionTreeAlgorithmClassifier

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

        elif sys.argv[1] == 'random_forest':
            model = RandomForestAlgorithmClassifier()

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

        elif sys.argv[1] == 'decision_tree':
            model = DecisionTreeAlgorithmClassifier()


if __name__ == "__main__":
    main()
