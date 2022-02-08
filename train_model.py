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
    else:
        raise NotImplementedError(f'{model_name} has not been implemented')
    return  numeric_transformer

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


def train_model():
    local_logger.info('Train: Preparing data')
    df = pd.read_csv(train_data_path)
    y = df[target_var]
    df.drop(target_var, axis=1, inplace=True)
    df.drop('facility_type', axis=1, inplace=True)
    X = df
    local_logger.info('Creating pipeline')
    reg = create_pipeline_random_forest()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    reg.fit(X_train, y_train)
    param_grid = {
        "regressor__n_estimators": [10, 50, 100],
    }
    grid_search = GridSearchCV(reg, param_grid, cv=10)
    return grid_search.best_params_
