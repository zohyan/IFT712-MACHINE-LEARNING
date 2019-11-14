import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.perceptron import PerceptronClassifier
from classifiers.random_forest import RandomForestAlgorithmClassifier
from classifiers.svm import SVMClassifier
from classifiers.fully_connected import FullyConnectedClassifier
from classifiers.adaboost import AdaBoostAlgorithmClassifier
from classifiers.decision_tree import DecisionTreeAlgorithmClassifier

def main():

    if sys.argv[1] == 'logistic_regression':
        model = LogisticRegressionClassifier()
        model.train()
        model.evaluate(training=True, metrics="Accuracy")
        model.evaluate(training=False, metrics="Accuracy")

    if sys.argv[1] == 'perceptron':
        model = PerceptronClassifier()
        model.train()
        model.evaluate(training=True, metrics="Accuracy")
        model.evaluate(training=False, metrics="Accuracy")

    if sys.argv[1] == 'random_forest':
        model = RandomForestAlgorithmClassifier()
        model.train()
        model.evaluate(training=True, metrics="Accuracy")
        model.evaluate(training=False, metrics="Accuracy")

    if sys.argv[1] == 'svm':
        model = SVMClassifier()
        model.train()
        model.evaluate(training=True, metrics="Accuracy")
        model.evaluate(training=False, metrics="Accuracy")

    if sys.argv[1] == 'fully_connected':
        model = FullyConnectedClassifier()
        model.train()
        model.evaluate(training=True, metrics="Accuracy")
        model.evaluate(training=False, metrics="Accuracy")

    if sys.argv[1] == 'adaboost':
        model = AdaBoostAlgorithmClassifier()
        model.train()
        model.evaluate(training=True, metrics="Accuracy")
        model.evaluate(training=False, metrics="Accuracy")

    if sys.argv[1] == 'decision_tree':
        model = DecisionTreeAlgorithmClassifier()
        model.train()
        model.evaluate(training=True, metrics="Accuracy")
        model.evaluate(training=False, metrics="Accuracy")

if __name__ == "__main__":
    main()