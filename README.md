# IFT712-MACHINE-LEARNING

My project of session in the Machine Learning course at the University of Sherbrooke. It is done in a team of three people.


## Requirements

The project was developed on python 3.6 on Ubuntu 18.04. So for running the project make sure you have a python version 3.x and it is better to run it on linux.

## Dependencies

Our code uses libraries like :
* scikit-learn
* Numpy
* Pandas
* Matplotlib

## Code design

Our code meets the standards imposed by pep8. We used the most possible design pattern to know the refactoring to write clean codes. The class structure is as follows :

- __IFT712\-MACHINE\-LEARNING__
   - __classifiers__
     - [abstract\_classifier.py](classifiers/abstract_classifier.py)
     - [adaboost.py](classifiers/adaboost.py)
     - [bagging.py](classifiers/bagging.py)
     - [decision\_tree.py](classifiers/decision_tree.py)
     - [fully\_connected.py](classifiers/fully_connected.py)
     - [logistic\_regression.py](classifiers/logistic_regression.py)
     - [random\_forest.py](classifiers/random_forest.py)
     - [svm.py](classifiers/svm.py)
   - __cross\_validation__
     - [cross\_validation.py](cross_validation/cross_validation.py)
   - __data\_utils__
     - [data\_loader.py](data_utils/data_loader.py)
     - [data\_preprocessing.py](data_utils/data_preprocessing.py)
     - [data\_visualization.py](data_utils/data_visualization.py)
   - __datasets__
     - [gender\_submission.csv](datasets/gender_submission.csv)
     - [test.csv](datasets/test.csv)
     - [train.csv](datasets/train.csv)
   - __metrics__
     - [metrics.py](metrics/metrics.py)
   - __model\_summary__
     - [model\_summary.py](model_summary/model_summary.py)
   - [main.py](main.py)


## Running

There are two ways to test our project:

* Either pick one model to test :

```console
$ python3 main.py <model> <metrics> <cross_validation> <data_preprocessing>

model : adaboost | logistic_regression | random_forest | svm | fully_connected | decision_tree | bagging
metrics : accuracy | confusion_matrix | roc
cross_validation : 0 | 1
data_preprocessing : 0 | 1
```
* Or cross validate all models to compare the results by displaying a histogram :

```console
$ python3 model_summary/model_summary.py <data_preprocessing>

data_preprocessing : 0 | 1
```

## Writing

We also write a [report](Rapport_de_projet_Machine_Learning.pdf) that comes with the code.
