from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionGscvModel(BaseModel):

    def __init__(self, param_grid, cv, scoring):

        # definizione di grid search con cross validation
        # estimator -> modello di regressione logistica
        # param_grid -> insieme di parametri della grid search
        # cv -> determina la strategia di split della cross validation (parametro k)
        # scoring -> strategia per valutare le prestazioni del modello (valutiamo l'accuracy)
        self.model = GridSearchCV(estimator=LogisticRegression(random_state=42, max_iter=5000), 
                                  param_grid=param_grid, 
                                  cv=cv,
                                  scoring=scoring)
        
    def best_estimator(self):
        return self.model.best_estimator_
    
    def print_best_params(self):
        return self.model.best_params_

    def predict(self, X_test):
        return self.best_estimator().predict(X_test)