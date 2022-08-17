from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, classification_report
import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


class Classification:
    def __init__(self, path):
        self.path = path
        self.read_data()
        self.test_split()
        self.scaler = RobustScaler()

    def test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=77)

    def read_data(self):
        df = pd.read_csv(self.path)
        df = df[['Diabetes_binary', 'HighChol', 'BMI', 'Smoker', 'Stroke',
                 'HeartDiseaseorAttack', 'GenHlth', 'DiffWalk', 'Sex', 'Age', 'MentHlth']]
        self.X = df.drop(['Diabetes_binary'], axis=1).to_numpy()
        self.y = df['Diabetes_binary'].to_numpy()

    def roc(self, model_name):
        logit_roc_auc = roc_auc_score(self.y_test, model_name.predict(self.X_test))
        fpr, tpr, thresholds = roc_curve(self.y_test, model_name.predict_proba(self.X_test)[:,1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{str(model_name).split('(')[0]}(area = {round(logit_roc_auc,2)})")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()

    def report_results(self, classifier, y_pred):
        print(f"Accuracy/Score: {accuracy_score(self.y_test, y_pred)}")
        #print(f"Precision = {precision_score(self.y_test, y_pred)}\n")
        #print(f"Confusion Matrix:\n {confusion_matrix(self.y_test, y_pred)}")
        #print(plot_confusion_matrix(classifier, self.X_test, self.y_test, values_format="d", cmap='Blues'))
        #print(classification_report(self.y_test, y_pred, digits=4))
        #self.roc(classifier)

    def fit_predict_score(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.report_results(model, y_pred)

    def random_subspaces(self, model):
        bgclassifier = BaggingClassifier(base_estimator=model,
                                         n_estimators=500,
                                         max_samples=1.0,
                                         bootstrap=False,
                                         max_features=0.5,
                                         bootstrap_features=True,
                                         random_state=42)
        self.fit_predict_score(bgclassifier)

    def pasting(self, model):
        bgclassifier = BaggingClassifier(base_estimator=model,
                                         n_estimators=500,
                                         max_samples=0.3,
                                         bootstrap=False,
                                         n_jobs=-1,
                                         random_state=42)
        self.fit_predict_score(bgclassifier)

    def bagging_oob(self, model):
        bgclassifier = BaggingClassifier(base_estimator=model,
                                         n_estimators=500,
                                         max_samples=0.25,
                                         bootstrap=True,
                                         oob_score=True,
                                         random_state=42)
        bgclassifier.fit(self.X_train, self.y_train)
        print(f"Score: {bgclassifier.oob_score_}")

    def bagging(self, model):
        bgclassifier = BaggingClassifier(base_estimator=model,
                                         n_estimators=500,
                                         max_samples=0.25,
                                         bootstrap=True,
                                         random_state=42)
        self.fit_predict_score(bgclassifier)

    def fit_predict_score(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.report_results(model, y_pred)

    def svc_best_params(self):
        # fit and transform both train and test
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        max_accur = 0.0
        for k in range(3, 20, 2):
            knn1 = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            knn1.fit(self.X_train, self.y_train)
            accur = knn1.score(self.X_test, self.y_test)
            if accur > max_accur:
                max_accur = accur
                best_k = k
            #print(
            #    f"Accuracy for k= {k} is {accur} Cross Validation value is \
            #    {np.mean(cross_val_score(knn1, self.X_test, self.y_test, cv=5))}")
        return best_k

    def run_all_models(self, model):
        print(f"{model.__class__.__name__} Model results:")
        self.fit_predict_score(model)
        print(f"{model.__class__.__name__} Bagging model results:")
        self.bagging(model)
        print(f"{model.__class__.__name__} Bagging_OOB model result:")
        self.bagging_oob(model)
        print(f"{model.__class__.__name__} Pasting model result:")
        self.pasting(model)
        print(f"{model.__class__.__name__} Boosting: adaBoost model result:")
        ada_reg = AdaBoostClassifier(model,
                                     n_estimators=300,
                                     learning_rate=0.5)
        self.fit_predict_score(ada_reg)
        print(f"----  End of {model.__class__.__name__} Model -----")

    def run_regression(self):
        self.knn_solve()
        self.svc_solve()
        self.dt_solve()

    def knn_solve(self):
        k = self.svc_best_params()
        knn = KNeighborsClassifier(n_neighbors=k)
        print("----- Starting KNeighborsClassifier Model -----")
        self.run_all_models(knn)

    def svc_solve(self):
        # Select the parameters
        # Defining the space of the parameters, using the dictionary method
        parameters = {
            'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
            'C': [1, 10],
            'gamma': ['scale', 'auto']
        }
        # Setting up the GridSearchCV and sending the support vector values and parameter dictionary
        # Select cross validation whose default is 5
        svm11 = GridSearchCV(SVC(), parameters, cv=5)
        # Performance evaluation is performed using the cross validation method
        svm11.fit(self.X_train1, self.y_train1)
        cv_results = pd.DataFrame(svm11.cv_results_)
        # cv_results[['param_C', 'param_gamma', 'param_kernel', 'mean_test_score', 'rank_test_score']]
        # Selected into the Classifier the best results for the parameters
        svc = SVC(kernel=svm11.best_params_['kernel'],
                  C=svm11.best_params_['C'],
                  gamma=svm11.best_params_['gamma'],
                  probability=True)
        print("----- Starting SVC Model -----")
        self.run_all_models(svc)

    def dt_solve(self):
        dt_clf = DecisionTreeClassifier(criterion="gini", random_state=50)
        self.test_split()
        print("----- Starting Decision Tree Model -----")
        self.run_all_models(dt_clf)

    def nb_solve(self):
        gnb_clf = GaussianNB()
        pipeline = make_pipeline(RobustScaler(),
                                 GaussianNB())