from alex_func import *
from train_model import *
from test_train_data import *

if __name__ == '__main__':
    # -- Get data
    extract_data()
    train_df = load_train_data()
    test_df = load_test_data()
    # -- Manipulate data
    train_df, test_df = group_categ_var(
        dt_train_original=train_df,
        dt_test_original=test_df,
        categ_var=[i for i in train_df.columns if train_df.dtypes[i]=='object'],
        groupby_var=[],
        tgt_var="site_eui",
        groupby_fun=["mean", "median"]
    )
    # -- train model
    train_model(dt_train_original=train_df, model_name='random_forest')
