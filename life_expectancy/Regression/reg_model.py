import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
from model_helper import Model
from sklearn.model_selection import GridSearchCV


# Class to handle the Regression functions
class RegModel(Model):
    def __init__(self, data, model, params=None):
        super().__init__(data, model, params)
        self.r_score = 0

    def search_and_fit(self):
        search = GridSearchCV(self.z_model, self.params, scoring='r2', cv=10)
        search.fit(self.X_train, self.y_train)
        return self.model(**search.best_params_)

    def set_x_y(self, data):
        # set the X,y parameters
        return data.drop('Life expectancy', axis=1), data['Life expectancy']

    def get_scores(self):
        return r2_score(self.y_train, self.y_train_pred), r2_score(self.y_test, self.y_test_pred)

    def report_results(self):
        print(f"training score: {self.scores[0]}")
        print(f"testing score= {self.scores[1]}")
        print(f"Diff between train-test: {self.scores[0] - self.scores[1]}")
        print(f"Mean Absolute Error: {mean_absolute_error(self.y_test, self.y_test_pred)}")
        self.residual()

    def residual(self):
        # Plotting graphs
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].set_title("Residual Plot of Train samples")
        sns.distplot((self.y_train, self.y_train_pred), hist=False, ax=ax[0])
        ax[0].set_xlabel('y_train - y_train_pred')
        #Y_test VS. Y_train scatter plot
        ax[1].set_title('y_test VS. y_test_pred')
        ax[1].scatter(x=self.y_test, y=self.y_test_pred)
        ax[1].set_xlabel('y_test')
        ax[1].set_ylabel('y_test_pred')
        plt.show()
