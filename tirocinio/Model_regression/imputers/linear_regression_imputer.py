from .imputer_regression import ImputerRegression
from sklearn.linear_model import LinearRegression

class LinearRegressionImputer(ImputerRegression):

    def get_model(self):
        
        model = LinearRegression()
        return model