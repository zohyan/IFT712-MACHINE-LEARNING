import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sklearn.linear_model import LogisticRegression
from data_utils.data_preprocessing import DataPreprocessing
from metrics.metrics import Metrics

class LogisticRegressionClassifier:

    def __init__(self):
        self.logreg = LogisticRegression()
        self.metrics = Metrics()
        self.X_train, self.Y_train, self.X_test, self.Y_test = DataPreprocessing().naive_preprocessing_data()

    def train(self):
        self.logreg.fit(self.X_train, self.Y_train)

    def predict(self, x):
        self.train()
        return self.logreg.predict(x)

    def evaluate(self, training=True, metrics="Accuracy"):
        if training:
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        if metrics == "Accuracy":
            train_acc_logreg = round(self.logreg.score(x, y) * 100, 2)
            print("Train accuracy", train_acc_logreg, " %")

        elif metrics == "Log Loss":
            print("Log loss", round(log_loss(self.predict(x), y), 2))

        elif metrics == "Confusion Matrix":
            print(confusion_matrix(self.predict(x), y, labels=[0, 1]))
            plt.imshow(confusion_matrix(y, self.predict(x)), interpolation='nearest', cmap=plt.cm.Blues)
            s = [['TN', 'FP'], ['FN', 'TP']]
            plt.ylabel("True Values")
            plt.xlabel("Predicted Values")
            plt.xticks([])
            plt.yticks([])
            for i in range(confusion_matrix(y, self.predict(x)).shape[0]):
                for j in range(confusion_matrix(y, self.predict(x)).shape[1]):
                    plt.text(j, i, str(s[i][j]) + " = " + str(confusion_matrix(y, self.predict(x))[i, j]), ha="center", va="center")
            plt.title("CONFUSION MATRIX VISUALIZATION")
            plt.show()

LR = LogisticRegressionClassifier()
LR.train()
LR.evaluate(training=False, metrics="Confusion Matrix")