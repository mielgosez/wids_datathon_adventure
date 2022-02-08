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


if __name__ == '__main__':
    local_df = load_train_data()
    histogram_target_data(df=local_df)
    correlation_target_energy_star(df=local_df)
    correlations = local_df.corr()
    pass
