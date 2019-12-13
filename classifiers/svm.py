import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.svm import SVC
from classifiers.abstract_classifier import AbstractClassifier


class SVMClassifier(AbstractClassifier):

    def __init__(self, kernel='rbf', C=1, gamma=1, mode='0'):
        super().__init__(SVC(probability=True, kernel=kernel, gamma=gamma, C=C), mode)