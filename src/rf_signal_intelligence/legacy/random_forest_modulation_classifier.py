"""Legacy RandomForest modulation classifier used by archived notebooks."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForestModulationClassifier:
    """Minimal scikit-learn style wrapper retained for archived experiments."""

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
