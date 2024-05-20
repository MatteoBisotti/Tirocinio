from sklearn.linear_model import LogisticRegression
from .imputer_regression import ImputerRegression

class LogisticRegressionImputer(ImputerRegression):

    def get_model(self):

        model = LogisticRegression()
        return model