# Base Python
from logging import Logger
import zipfile
# Data Science
import pandas as pd
import kaggle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
# Custom
from data.metadata import *


local_logger = Logger(name='kaggle')


def extract_data():
    local_logger.info('Extracting data')
    kaggle.api.competition_download_files(competition='widsdatathon2022',
                                          path='./data',
                                          force=True)
    local_logger.info('Extracting files from zip in data folder')
    with zipfile.ZipFile(os.path.join(data_path, 'widsdatathon2022.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_path)
    local_logger.info('Removing zip file')
    os.remove(os.path.join(data_path, 'widsdatathon2022.zip'))


def create_preprocessing_numeric_pipeline(model_name: str):
    if model_name == 'random_forest':
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
    elif model_name == 'XGBoost':
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
    else:
        raise NotImplementedError(f'{model_name} has not been implemented')
    return numeric_transformer


def create_preprocessing_pipeline(model_name: str):
    numeric_transformer = create_preprocessing_numeric_pipeline(model_name=model_name)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, incomplete_variables),
            ("cat", categorical_transformer, categorical_variables),
        ]
    )
    return preprocessor


def create_pipeline_random_forest():
    preprocessor_pipeline = create_preprocessing_pipeline(model_name='random_forest')
    reg = Pipeline(
        steps=[("preprocessor", preprocessor_pipeline), ("regressor", RandomForestRegressor())]
    )
    return reg


def create_pipeline_XGBoost():
    preprocessor_pipeline = create_preprocessing_pipeline(model_name='XGBoost')
    reg = Pipeline(
        steps=[("preprocessor", preprocessor_pipeline), ("regressor", XGBRegressor())]
    )
    return reg


def train_model(dt_train_original: pd.DataFrame, model_name: str):
    df = dt_train_original.copy()
    local_logger.info('Train: Preparing data')
    y = df[target_var]
    df.drop(target_var, axis=1, inplace=True)
    df.drop(['facility_type','id'], axis=1, inplace=True)
    X = df
    local_logger.info('Creating pipeline')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    if model_name == 'random_forest':
        reg = create_pipeline_random_forest()
        param_grid = {
            "regressor__n_estimators": [10, 50, 100],
        }
    elif model_name == 'XGBoost':
        reg = create_pipeline_XGBoost()
        param_grid = {'regressor__max_depth': [3, 6, 10],
                  'regressor__learning_rate': [0.01, 0.05, 0.1],
                  'regressor__n_estimators': [50, 100]}
    else:
        raise NotImplementedError(f'{model_name} has not been implemented')
    grid_search = GridSearchCV(reg, param_grid, cv=10, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search
