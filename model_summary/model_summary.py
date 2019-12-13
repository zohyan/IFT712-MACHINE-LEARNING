import os, sys
sys.path.append(os.path.dirname(os.path.join("..")))
import numpy as np
from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.random_forest import RandomForestAlgorithmClassifier
from classifiers.svm import SVMClassifier
from classifiers.fully_connected import FullyConnectedClassifier
from classifiers.adaboost import AdaBoostAlgorithmClassifier
from classifiers.decision_tree import DecisionTreeAlgorithmClassifier
from classifiers.bagging import BaggingAlgorithmClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    mode = sys.argv[1]

    models = [
        ('logistic_regression', LogisticRegressionClassifier(mode=mode)),
        ('random_forest', RandomForestAlgorithmClassifier(mode=mode)),
        ('svm', SVMClassifier(mode=mode)),
        ('fully_connected', FullyConnectedClassifier(mode=mode)),
        ('adaboost', AdaBoostAlgorithmClassifier(mode=mode)),
        ('decision_tree', DecisionTreeAlgorithmClassifier(mode=mode)),
        ('bagging', BaggingAlgorithmClassifier(mode=mode))
    ]

    accuracy = []

    hyperparameters = {
        'logistic_regression': {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(0, 4, 10)
        },
        'random_forest': {
            'n_estimators': [200, 300, 400],
            'max_features': ['auto'],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 3],
            'bootstrap': [True]
        },
        'svm': {
                'C': [0.1, 1, 10],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel': ['rbf']
        },
        'fully_connected': {
            'hidden_layer_sizes': [(5,), (5, 5)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [1e-5, 3e-4],
            'learning_rate_init': [1e-2, 1e-4]
        },
        'adaboost': {
            'base_estimator': [DecisionTreeAlgorithmClassifier().model],
            'n_estimators': [50, 60, 70],
            'algorithm': ['SAMME.R', 'SAMME']
        },
        'decision_tree': {
            'criterion': ['entropy', 'gini'],
            'max_depth': [4]
        },
        'bagging': {
            'weights': [[int(x) for x in list("{0:0b}".format(i).zfill(4))] for i in range(1, 2 ** 2)]
        }
    }

    for name, model in models:
        accuracy.append(model.tunning_model(hyperparameters[name], 5, 'accuracy'))

    results = pd.DataFrame({
        'Model': [
            'Logistic Regression', 'Random Forest',
            'SVM', 'Fully Connected', 'Adaboost','Decision Tree',
            'Bagging Model'
        ],
        'Score': accuracy
    })

    result_df = results.sort_values(by='Score', ascending=False)

    bestmodelgraph = result_df.head(7)

    plt.rcParams["figure.figsize"] = (30, 30)

    ax = sns.barplot(x="Score", y="Model", data=bestmodelgraph, palette='Blues_d')

    plt.show()


if __name__ == "__main__":
    main()