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

        plt.title("CONFUSION MATRIX VISUALIZATION OF THE "+label)
        plt.show()

    def confusion_matrix_after_validation(self, pred, y):
        plt.imshow(confusion_matrix(y, pred), interpolation='nearest', cmap=plt.cm.Blues)
        plt.ylabel("True Values")
        plt.xlabel("Predicted Values")
        plt.xticks([])
        plt.yticks([])

        for i in range(confusion_matrix(y, pred).shape[0]):
            for j in range(confusion_matrix(y, pred).shape[1]):
                plt.text(j, i, str([['TN', 'FP'], ['FN', 'TP']][i][j]) + " = " + str(confusion_matrix(y, pred)[i, j]), ha="center", va="center")

        plt.title("CONFUSION MATRIX VISUALIZATION AFTER VALIDATION ")
        plt.show()

    def plot_roc(self, model, x, y, label):
        prob = model.predict_proba(x)
        prob = prob[:, 1]
        auc = roc_auc_score(y, prob)
        print(label+' AUC: %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(y, prob)
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def plot_roc_after_validation(self, prob, y):
        auc = roc_auc_score(y, prob)
        print('AUC after cross-validation : %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(y, prob)
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()