import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.ensemble import AdaBoostClassifier
from classifiers.abstract_classifier import AbstractClassifier


class AdaBoostAlgorithmClassifier(AbstractClassifier):

    def __init__(self):
        super().__init__(AdaBoostClassifier(n_estimators=50, learning_rate=1))
