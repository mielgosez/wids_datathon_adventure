import pandas as pd
from utils import load_train_data
import matplotlib.pyplot as plt
from data.metadata import *


def histogram_target_data(df: pd.DataFrame):
    df[target_var].plot.hist(bins=100)
    plt.show()


def correlation_target_energy_star(df: pd.DataFrame):
    df.plot.scatter(x='energy_star_rating',
                    y=target_var,
                    alpha=0.05)
    plt.xlabel('Energy Star Rating')
    plt.ylabel(target_var)
    plt.show()


def impute_energy_start_rating(df: pd.DataFrame):
    df.drop(target_var, axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    df.drop('facility_type', axis=1, inplace=True)
    target_name = 'energy_star_rating'
    incomplete_variables.remove(target_name)
    df.drop(incomplete_variables, axis=1, inplace=True)
    df.dropna(inplace=True)
    y = df[target_name]
    df.drop(categorical_variables, axis=1, inplace=True)
    print(df.size)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    pca = PCA(.98)
    X = pca.fit_transform(X=df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    param_grid = {
        "n_estimators": [10, 50, 100],
    }
    reg = RandomForestRegressor()
    grid_search = GridSearchCV(reg, param_grid, cv=10)
    grid_search.fit(X_train, y_train)
    grid_search.transform(X_test)
    return -1


if __name__ == '__main__':
    local_df = load_train_data()
    impute_energy_start_rating(df=local_df)
    histogram_target_data(df=local_df)
    correlation_target_energy_star(df=local_df)
    correlations = local_df.corr()
    pass
