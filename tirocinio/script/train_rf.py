import sys
sys.path.append('../base_lib')
import functions as func

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [3, 4, 5],
        'max_depth': [8],
        'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 4, 6],
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }

    grid_search = GridSearchCV(estimator=model,
                                param_grid=param_grid,
                                cv=5,
                                scoring='f1_macro')
    
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def main():
    dataset = func.load_csv()

    X = dataset.drop(['LUX_01'], axis=1)
    y = dataset['LUX_01']

    df = func.oversampling(dataset, X, y)

    dataset = func.drop_cols(dataset)
    df = func.drop_cols(df)

    training_set, testing_set = func.train_test(dataset, df, False)

    feature_cols = ['BREED', 'GENDER_01', 'AGEATSURGERYmo', 'BODYWEIGHTKG', 'Taglia', 'BCS', 
                    'YEAR', 'GENERATION', 'STEMSIZE', 'CUPSIZE', 'NECKSIZE', 'HEADSIZE', 'ALO', 'CUPRETROVERSION', 'STEMANTEVERSIONREAL', 
                    'RECTUSFEMORISM.RELEASE', 'LUX_CR']

    X_train = training_set[feature_cols]
    y_train = training_set['LUX_01']

    model = train_random_forest(X_train, y_train)
    return model