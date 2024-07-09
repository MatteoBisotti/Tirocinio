import sys
sys.path.append('../base_lib')
import functions as func

from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=8, 
                               min_samples_split=4, 
                              min_impurity_decrease=0.0, 
                              criterion='gini')
    
    model.fit(X_train, y_train)

    return model

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

    model = train_decision_tree(X_train, y_train)
    return model