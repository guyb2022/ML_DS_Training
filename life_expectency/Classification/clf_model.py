from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from model_helper import Model


# Class to handle the Classification functions
class ClfModel(Model):
    def __init__(self, data, model, params=None):
        super().__init__(data, model, params)

    def search_and_fit(self):
        search = GridSearchCV(self.z_model, self.params, scoring='accuracy', cv=10)
        search.fit(self.X_train, self.y_train)
        #print(f"Best Params: {search.best_params_}")
        return self.model(**search.best_params_)

    def set_x_y(self, data):
        # set the X,y parameters
        return data.drop('target', axis=1), data['target']

    def get_scores(self):
        return accuracy_score(self.y_train, self.y_train_pred), accuracy_score(self.y_test, self.y_test_pred)

    def roc(self, clf_model):
        logit_roc_auc = roc_auc_score(self.y_test, clf_model.predict(self.X_test))
        fpr, tpr, thresholds = roc_curve(self.y_test, clf_model.predict_proba(self.X_test)[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{self.model_name}(area = {round(logit_roc_auc,2)})")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()

    def report_results(self):
        print(f"Accuracy/Score: {accuracy_score(self.y_test, self.y_test_pred)}")
        print(f"Precision = {precision_score(self.y_test, self.y_test_pred)}\n")
        print(f"Confusion Matrix:\n {confusion_matrix(self.y_test, self.y_test_pred)}")
        plot_confusion_matrix(self.z_model, self.X_test, self.y_test, values_format="d", cmap='Blues')
        print(classification_report(self.y_test, self.y_test_pred, digits=4))
        self.roc(self.z_model)

