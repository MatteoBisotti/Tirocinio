from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer

class ImputerKNN(ABC):

    def impute_value(self, dataset):
        imputer = KNNImputer()
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9]
        }

        grid_searh = GridSearchCV(estimator=imputer, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_searh.fit(dataset)

        best_param = grid_searh.best_params_['n_neighbors']
        print("Miglior parametro n_neighbors:", best_param)

        imputer = KNNImputer(n_neighbors=best_param, weights='distance')
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

        return dataset