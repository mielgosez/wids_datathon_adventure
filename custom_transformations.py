from sklearn.base import BaseEstimator
from sklearn.pipeline import TransformerMixin
import xgboost as xgb


class ImputeFoggyDays(TransformerMixin, BaseEstimator):
    def __init__(self, categorical_variables: list):
        self.categorical_variables = categorical_variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        return X_


class ImputeFoggyDays(TransformerMixin, BaseEstimator):
    def __init__(self, categorical_variables: list):
        self.categorical_variables = categorical_variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        return X_
