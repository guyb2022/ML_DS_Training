from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
from os.path import exists


class Model:
    def __init__(self, data, model, params=None):
        self.data = data
        self.test_size = 0.2
        self.X, self.y = self.set_x_y(data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.test_split()
        self.z_model = model()
        self.model = model
        self.model_name = self.model().__class__.__name__
        self.path = 'models/' + self.model_name + ".sav"
        self.params = params
        self.test_score = 0
        self.train_score = 0
        self.best_score = 0
        self.scores = 0

    def test_split(self):
        # Create the train-test data
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=77)

    def normalize_data(self):
        # Run a standard normalization
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def prepare_data(self, data):
        # Prepare the data for the model
        self.X, self.y = self.set_x_y(data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.test_split()
        self.normalize_data()

    def load_pickle(self):
        # load the model from file
        loaded_model = pickle.load(open(self.path, 'rb'))
        return loaded_model

    def save_model(self, model):
        # save the model to disk
        pickle.dump(model, open(self.path, 'wb'))

    def search_and_fit(self):
        search = GridSearchCV(self.z_model, self.params, scoring='roc_auc', cv=10)
        search.fit(self.X_train, self.y_train)
        #print(f"Best Params: {search.best_params_}")
        return self.model(**search.best_params_)

    def fit_predict_score(self, data, show=True, opt=False, lpick=True, spick=True):
        # Helper class to handle all the function needed to run the model
        # Main function: fit_predict_score(data, show=True, opt=False, lpick=False, spick=True)
        # 1. Run clean model without saving:           fit_predict_score(df,1,0,0,0)
        # 2. Run clean model and save model to file:   fit_predict_score(df,1,0,0,1)
        # 3. Load model and save model to file:        fit_predict_score(df,1,0,1,1)
        # 3. Run optimization without showing results: fit_predict_score(df,0,1,0,0)

        # In case we are running on regular mode, we would like to use
        # the params tuning/load mode
        print("Preparing the data")
        # In case opt was used we need to prepare the data after each iteration
        self.prepare_data(data)
        if not opt:
            # Check if model exist in models/model_name.sav
            if exists(self.path) and lpick:
                # load pickled saved model
                print("Searching for saved pickle model file")
                print(f"Loading model {self.model_name}.sav from file")
                self.z_model = self.load_pickle()
            elif not exists(self.path) or not lpick:
                print("Running on a new model")
                self.z_model = self.model()
            if self.params:
                # Params was assigned to the model
                print("Finding best params with GridSearch")
                self.z_model = self.search_and_fit()
        # Fit & Predict the chosen model
        print("Done preprocessing procedures -> Fit & Predict")
        self.z_model.fit(self.X_train, self.y_train)
        # Save model to pickle file, if not in optimization mode.
        if not opt and spick:
            print(f"Saving new model to file: models/{self.model_name}.sav")
            self.save_model(self.z_model)
        # Predict
        self.y_test_pred = self.z_model.predict(self.X_test)
        self.y_train_pred = self.z_model.predict(self.X_train)
        # Get scores for train & test
        self.scores = self.get_scores()
        # Show results
        if show:
            self.report_results()
        return self.scores

    def feature_extraction(self, data):
        # UNI VARIATE SELECTION
        # Feature Extraction with Uni variate Statistical Tests (f_regression)
        # load data
        X, y = self.set_x_y(data)
        names = pd.DataFrame(X.columns)
        model = SelectKBest(score_func=f_regression, k=4)
        results = model.fit(X, y)
        results_df = pd.DataFrame(results.scores_)
        # Concat results_df and name columns
        scored = pd.concat([names, results_df], axis=1)
        scored.columns = ["Feature", "Score"]
        scored.sort_values(by=['Score'], ascending=False)
        final_columns = scored[scored.Score > 0]
        df_sol = final_columns.sort_values(by=['Score'], ascending=True)
        return df_sol

    def feature_reduction(self, data):
        # Perform the feature reduction by the corr() values
        df_new = data.copy()
        to_drop = self.feature_extraction(df_new).iloc[0]['Feature']
        print(f"Checking to drop: {to_drop}")
        df_new.drop(to_drop, axis=1, inplace=True)
        # train/test/split again after dropping column
        _, new_test_score = self.fit_predict_score(df_new, show=False, opt=True, lpick=False, spick=False)
        print(f"new_test_score: {new_test_score} --- self.best_score: {self.best_score}")
        if new_test_score < self.best_score:
            return 0, data, new_test_score
        return 1, df_new, new_test_score

    def optimize_model(self):
        """
        Run the optimization until no progress is reached
        """
        done = False
        new_df = self.data.copy()
        while not done:
            result, new_df, new_score = self.feature_reduction(new_df)
            if result == 0:
                print("Abort deleting feature, optimization completed")
                done = True
            else:
                self.best_score = new_score
        return new_df
