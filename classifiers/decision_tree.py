import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.tree import DecisionTreeClassifier
from classifiers.abstract_classifier import AbstractClassifier


class DecisionTreeAlgorithmClassifier(AbstractClassifier):

    def __init__(self):
        model = DecisionTreeClassifier()
        super().__init__(model)