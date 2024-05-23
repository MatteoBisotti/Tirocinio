from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

class DecisionTreeModel(BaseModel):

    def __init__(self, max_depth):
        self.model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)

    def print_tree(self, feature_cols):
        plt.figure(figsize=(16, 12))
        plot_tree(decision_tree=self.model, 
                feature_names=feature_cols, 
                filled=True, 
                rounded=True, 
                class_names=True, max_depth=2)
        plt.title("Albero di decisione")
        plt.show()

    def feature_importance(self):
        return self.model.feature_importances_

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