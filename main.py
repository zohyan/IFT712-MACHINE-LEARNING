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

    if sys.argv[2] in ['accuracy', 'log_loss', 'confusion_matrix', 'hinge_loss', 'roc']:

        if sys.argv[1] == 'logistic_regression':
            model = LogisticRegressionClassifier()
            model.train()
            model.evaluate(training=True, metrics=sys.argv[2])
            model.evaluate(training=False, metrics=sys.argv[2])

            if sys.argv[3] == '1':
                hyperparameters = dict(penalty=['l1', 'l2'], C=np.logspace(0, 4, 10))
                model.tunning_model(hyperparameters, 5, sys.argv[2])

        elif sys.argv[1] == 'perceptron':
            model = PerceptronClassifier()

        elif sys.argv[1] == 'random_forest':
            model = RandomForestAlgorithmClassifier()

        elif sys.argv[1] == 'svm':
            model  = SVMClassifier()

            model.train()
            model.evaluate(training=True, metrics=sys.argv[2])
            model.evaluate(training=False, metrics=sys.argv[2])

            if sys.argv[3] == '1':
                Cs = [0.1, 1, 10, 100, 1000]
                gammas = [1, 0.1, 0.01, 0.001, 0.0001]
                kfold = 5
                hyperparameters = {'C': Cs, 'gamma': gammas , 'kernel':  ['rbf']}

                model.tunning_model(hyperparameters, kfold, sys.argv[2])

        elif sys.argv[1] == 'fully_connected':
            model = FullyConnectedClassifier()

        elif sys.argv[1] == 'adaboost':
            model = AdaBoostAlgorithmClassifier()

        elif sys.argv[1] == 'decision_tree':
            model = DecisionTreeAlgorithmClassifier()


if __name__ == "__main__":
    main()