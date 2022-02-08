from data.metadata import *
import pandas as pd
import kaggle


def load_train_data():
    return pd.read_csv('./data/train.csv')


def submit_results(df: pd.DataFrame,
                   message: str = 'Submission from Adventure team',
                   competition: str = competition_name):
    submission_file = './data/results/tmp_submission.csv'
    df.to_csv(submission_file, sep=',', decimal='.', index=False)
    kaggle.api.competition_submit(file_name=submission_file,
                                  message=message,
                                  competition=competition)
    os.remove(submission_file)
