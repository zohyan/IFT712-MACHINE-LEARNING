import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.ensemble import RandomForestClassifier
from classifiers.abstract_classifier import AbstractClassifier


class RandomForestAlgorithmClassifier(AbstractClassifier):

    def __init__(self):
        model = RandomForestClassifier(n_estimators=100)
        super().__init__(model)
