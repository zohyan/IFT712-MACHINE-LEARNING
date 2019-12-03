import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.linear_model import LogisticRegression
from classifiers.abstract_classifier import AbstractClassifier


class LogisticRegressionClassifier(AbstractClassifier):

    def __init__(self, penalty='l1', C=1):
        super().__init__(LogisticRegression(penalty=penalty, solver="liblinear", C=C))
