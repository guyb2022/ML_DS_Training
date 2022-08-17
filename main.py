import warnings
from helper_functions import Classification
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    path = 'data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
    hlp = Classification(path)
    hlp.run_regression()


