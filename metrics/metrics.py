import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Metrics:

    def accuracy(self, model, x, y, label):
        accuracy = round(model.score(x, y) * 100, 2)
        print(label + ' accuracy', accuracy, " %")


    def accuracy_after_validation(self, pred, y):
        print('Testing accuracy after cross-validation ', round(metrics.accuracy_score(pred, y) * 100), " %")

    def confusion_matrix(self, model, x, y, label):
        plt.imshow(confusion_matrix(y, model.predict(x)), interpolation='nearest', cmap=plt.cm.Blues)
        plt.ylabel("True Values")
        plt.xlabel("Predicted Values")
        plt.xticks([])
        plt.yticks([])

        for i in range(confusion_matrix(y, model.predict(x)).shape[0]):
            for j in range(confusion_matrix(y, model.predict(x)).shape[1]):
                plt.text(j, i, str([['TN', 'FP'], ['FN', 'TP']][i][j]) + " = " + str(confusion_matrix(y, model.predict(x))[i, j]), ha="center", va="center")

        plt.title("CONFUSION MATRIX VISUALIZATION OF THE "+label.upper())
        plt.show()
