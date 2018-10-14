from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


class EGCoint(object):

    def __init__(self, x, y, on, col_name):
        self.x, self.y = EGCoint.clean_data(x, y, on, col_name)
        self.alpha = None
        self.beta  = None
        self.resid_mean = None
        self.resid_std  = None
        self.cl = None

    @classmethod
    def clean_data(cls, x, y, on, col_name):
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_df = pd.merge(left=x, right=y, on=on, how='outer')
        clean_df  = merged_df.loc[merged_df.notnull().all(axis=1), :]
        return clean_df[col_name + '_x'].values.reshape(-1,1), clean_df[col_name + '_y'].values.reshape(-1,1)

    def cal_spread(self, x, y):
        resid = y - (self.alpha + x * self.beta)
        norm_resid = (resid - self.resid_mean)/self.resid_std
        return norm_resid

    def get_sample(self, start, end):
        assert start < end and end < len(self.x), 'Error:Invalid Indexing.'
        x_sample = self.x[start:end]
        y_sample = self.y[start:end]
        return x_sample, y_sample

    def test_coint(self, x, y, cl):
        try:
            _, p_val, _ = coint(x, y)
            return p_val if p_val < cl else False
        except ValueError:
            print('Exception encountered: ValueError.')
            return False
        except Exception as e:
            print('Exception encountered: {}.'.format(e))
            return False

    def run_ols(self, x, y):
        reg = LinearRegression().fit(x, y)
        self.alpha = reg.intercept_
        self.beta  = reg.coef_[0]
        resid = y - (self.alpha + self.beta * x)
        self.resid_mean = resid.mean()
        self.resid_std  = resid.std()

    def calibrate(self, start, end, cl):
        self.cl = cl
        x, y    = self.get_sample(start, end)
        p_val   = self.test_coint(x, y, cl)
        if p_val:
            self.run_ols(x, y)
            spread = self.cal_spread(x, y)
            return spread, p_val
        else:
            return None





