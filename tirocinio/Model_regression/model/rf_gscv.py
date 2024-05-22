from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestGscvModel(BaseModel):

    def __init__(self, param_grid, cv, scoring):
        self.model = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                                  param_grid=param_grid,
                                  cv=cv,
                                  scoring=scoring)

    def best_estimator(self):
        return self.model.best_estimator_
    
    def print_best_params(self):
        return self.model.best_params_
    
    def predict(self, X_test):
        return self.best_estimator().predict(X_test)