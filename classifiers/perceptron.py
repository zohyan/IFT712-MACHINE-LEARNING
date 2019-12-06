import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.linear_model import Perceptron
from classifiers.abstract_classifier import AbstractClassifier


class PerceptronClassifier(AbstractClassifier):

    def __init__(self):
        model = Perceptron(tol=0.20, max_iter=1000)
        super().__init__(model)