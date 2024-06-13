from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

    def get_stats(self, X_test, y_test):
        predictions = self.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        return accuracy, precision, recall, f1, roc_auc
    

    def statistics(self, X_test, y_test):
        predictions = self.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        # Stampa dei risultati
        print("Accuratezza:", accuracy)
        print("Sensibilità:", recall)
        print("Specificità:", precision)
        print("F1-score:", f1)
        print("ROC AUC:", roc_auc)