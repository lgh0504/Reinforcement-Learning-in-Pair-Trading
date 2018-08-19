import pandas as pd
import os


def save_csv(df: pd.DataFrame, ticker: str, path: str):
    file_path = os.path.join(path, ticker + '.csv')
    df.to_csv(file_path, index=False)