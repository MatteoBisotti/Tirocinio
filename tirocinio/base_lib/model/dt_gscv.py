from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class DecisionTreeGscvModel(BaseModel):

    def __init__(self, param_grid, cv, scoring):

        self.model = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                  param_grid=param_grid,
                                  cv=cv,
                                  scoring=scoring)

    def best_estimator(self):
        return self.model.best_estimator_
    
    def print_best_params(self):
        return self.model.best_params_

    def predict(self, X_test):
        return self.best_estimator().predict(X_test)
    
    def print_tree(self, feature_cols):
        plt.figure(figsize=(16, 12))
        plot_tree(decision_tree=self.best_estimator(), 
                feature_names=feature_cols, 
                filled=True, 
                rounded=True, 
                class_names=True, max_depth=2)
        plt.title("Albero di decisione")
        plt.show()

    def feature_importance(self):
        return self.best_estimator().feature_importances_

    # grafico feature importance
    def graph_feature_importance(self, feature_name):
        importance = self.feature_importance()

        feature_importance = pd.DataFrame({'Feature': feature_name, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])

        plt.title("Importanza delle feature")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()