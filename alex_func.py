import pandas as pd
import numpy as np
from utils import load_train_data
from train_model import extract_data
import matplotlib.pyplot as plt
from data.metadata import *

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)


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

def load_test_data():
    return pd.read_csv('./data/test.csv')

def group_categ_var(
        dt_train_original:pd.DataFrame,
        dt_test_original:pd.DataFrame,
        categ_var:list,
        groupby_var:list,
        tgt_var:list,
        groupby_fun:list
):
    # -- Make copies
    dt_train = dt_train_original.copy()
    dt_test = dt_test_original.copy()
    # -- Create new vars
    for col in categ_var:
        to_group_cols = [col] + groupby_var
        new_df = dt_train.groupby(to_group_cols)[tgt_var].agg(groupby_fun).reset_index()
        for f in groupby_fun:
            new_col = col + "_" + f
            new_df.rename(columns={f: new_col}, inplace=True)
        dt_train = dt_train.merge(
            new_df,
            on=to_group_cols,
            how="left"
        )
        dt_test = dt_test.merge(
            new_df,
            on=to_group_cols,
            how="left"
        )
    # -- Return output
    return dt_train, dt_test


if __name__ == '__main__':
    extract_data()
    train_df = load_train_data()
    test_df = load_test_data()
    print(train_df.shape)
    print(test_df.shape)
    train_df, test_df = group_categ_var(
        dt_train_original=train_df,
        dt_test_original=test_df,
        categ_var=[i for i in train_df.columns if train_df.dtypes[i]=='object'],
        groupby_var=[],
        tgt_var="site_eui",
        groupby_fun=["mean", "median"]
    )
    print(train_df.shape)
    print(test_df.shape)

