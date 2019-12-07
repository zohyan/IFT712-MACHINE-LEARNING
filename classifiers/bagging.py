import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(_file_))))
from classifiers.abstract_classifier import AbstractClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class BaggingAlgorithmClassifier(AbstractClassifier):

    def __init__(self):

        rf = RandomForestClassifier(n_estimators=100)
        svm = SVC(C=5, gamma="auto", probability=True)
        fc_tanh = MLPClassifier(hidden_layer_sizes=(10, 10,), activation='tanh', max_iter=5000)
        fc_relu = MLPClassifier(hidden_layer_sizes=(10, 10,), activation='relu', max_iter=5000)

        model = VotingClassifier(estimators=[
            ('Random Forests', rf),
            ('SVM', svm),
            ('FC_tanh', fc_tanh),
            ('FC_relu', fc_relu)
        ], voting='soft')

        super().__init__(model)