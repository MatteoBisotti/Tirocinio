from abc import ABC, abstractmethod
from sklearn.metrics import classification_report

class BaseModel(ABC):

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def get_report(self, X_test, y_test):
        predictions = self.predict(X_test=X_test)
        return classification_report(y_test, predictions)

    def print_report(self, X_test, y_test):
        print("Report di classificazione:")
        print(self.get_report(X_test, y_test))