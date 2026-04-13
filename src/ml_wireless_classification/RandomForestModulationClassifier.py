"""Legacy compatibility shim for archived RandomForest notebooks.

This lightweight class preserves import compatibility for historical notebooks.
It intentionally provides a minimal scikit-learn style interface.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForestModulationClassifier:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
