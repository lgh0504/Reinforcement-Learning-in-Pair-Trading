from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import pandas as pd
import numpy as np


class EGCoint:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.p = None
        self.cl = None
        self.alpha = None
        self.beta  = None
        self.resid_mean = None
        self.resid_std  = None

    @classmethod
    def create_pair(cls, x, y, on, col_name):
        x, y = cls.clean_pair(x, y, on, col_name)
        return cls(x, y)

    def run_coint(self, cl):
        self.cl = cl
        is_coint = self.test_coint(cl)
        if is_coint:
            self.run_ols()
            return True
        else:
            return False

    @classmethod
    def clean_pair(cls, x, y, on, col_name):
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_df = pd.merge(left=x, right=y, on=on, how='outer')
        clean_df  = merged_df.loc[merged_df.notnull().all(axis=1), :]
        return clean_df[col_name + '_x'], clean_df[col_name + '_y']

    def test_coint(self, cl):
        try:
            _, self.p, _ = coint(self.x, self.y)
            return True if self.p < cl else False
        except ValueError:
            print('Exception encountered: ValueError.')
            return False
        except Exception as e:
            print('Exception encountered: {}.'.format(e))
            return False

    def run_ols(self):
        X     = sm.add_constant(self.x)
        model = sm.OLS(self.y, X)
        est = model.fit()
        self.alpha = est.params[0]
        self.beta  = est.params[1]
        self.resid_mean = est.resid.mean()
        self.resid_std  = est.resid.std()

    def cal_norm_resid(self, x, y):
        resid = y - (self.alpha + x * self.beta)
        norm_resid = (resid - self.resid_mean)/self.resid_std
        return norm_resid
