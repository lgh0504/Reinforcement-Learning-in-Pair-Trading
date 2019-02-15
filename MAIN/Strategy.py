from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


class Cointegration(object):

    def __init__(self, x, y, on, col_name):
        self.x, self.y, self.timestamp = Cointegration.clean_data(x, y, on, col_name)
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
        return clean_df[col_name + '_x'].values.reshape(-1, 1), \
               clean_df[col_name + '_y'].values.reshape(-1, 1), \
               clean_df['date'].values

    def cal_spread(self, x, y):
        resid      = y - x * self.beta
        norm_resid = (resid - self.resid_mean)/self.resid_std
        return norm_resid

    def get_sample(self, start, end):
        assert start < end < len(self.x), 'Error:Invalid Indexing.'
        x_sample    = self.x[start:end]
        y_sample    = self.y[start:end]
        time_sample = self.timestamp[start:end]
        return x_sample, y_sample, time_sample

    @staticmethod
    def test_coint(x, y, cl):
        try:
            _, p_val, _ = coint(x, y)
            return p_val if p_val < cl else False
        except ValueError:
            print('Exception Encountered: ValueError.')
            return False
        except Exception as e:
            print('Exception Encountered: {}.'.format(e))
            return False

    def run_ols(self, x, y):
        reg = LinearRegression().fit(x, y)
        self.beta  = reg.coef_[0]
        resid = y - self.beta * x
        self.resid_mean = resid.mean()
        self.resid_std  = resid.std()

    def calibrate(self, start, end, cl):
        self.cl = cl
        x, y, _ = self.get_sample(start, end)
        p_val   = self.test_coint(x, y, cl)
        if p_val:
            self.run_ols(x, y)
            spread = self.cal_spread(x, y)
            return spread, p_val
        else:
            return None

    def gen_order(self, start, end, trade_th, stop_loss):
        assert stop_loss > trade_th, 'Error:Invalid Stop Loss Level.'

        x, y, time = self.get_sample(start, end)
        spread     = self.cal_spread(x, y)
        spread_t0  = spread[:-1]
        spread_t1  = spread[1:]
        t_t1       = time[1:]

        ind_buy  = np.logical_and(spread_t0 >= -trade_th, spread_t1 <= -trade_th).reshape(-1, )
        ind_sell = np.logical_and(spread_t0 <= trade_th, spread_t1 >= trade_th).reshape(-1, )
        ind_stop = np.logical_or(np.logical_and(spread_t0 >= -stop_loss, spread_t1 <= -stop_loss).reshape(-1, ),
                                 np.logical_and(spread_t0 <= stop_loss, spread_t1 >= stop_loss).reshape(-1, ))

        order = np.array([None] * len(t_t1))
        order[ind_buy]  = 'Buy'
        order[ind_sell] = 'Sell'
        order[ind_stop] = 'Stop'
        order[-1]       = 'Stop'

        ind_order = order != None
        time      = t_t1[ind_order]
        price     = spread_t1[ind_order]
        order     = order[ind_order]
        x         = x[ind_order]
        y         = y[ind_order]
        gross_exp = y + abs(x) * self.beta

        return time, price, order, gross_exp

    @staticmethod
    def gen_trade_record(time, price, order):
        if len(order) == 0:
            return None

        n_buy_sell  = sum(order != 'Stop')
        trade_time  = np.array([None] * n_buy_sell)
        trade_price = np.array([None] * n_buy_sell)
        close_time  = np.array([None] * n_buy_sell)
        close_price = np.array([None] * n_buy_sell)
        long_short  = np.array([None] * n_buy_sell)

        current_holding = 0
        j = 0

        for i in range(len(order)):
            sign_holding = int(np.sign(current_holding))
            if order[i] == 'Buy':
                close_pos                    = (sign_holding < 0) * -current_holding
                close_time[j  - close_pos:j] = time[i]
                close_price[j - close_pos:j] = price[i][0]
                trade_time[j]                = time[i]
                trade_price[j]               = price[i][0]
                long_short[j]                = 1
                buy_sell        = close_pos + 1
                current_holding = current_holding + buy_sell
                j += 1
            elif order[i] == 'Sell':
                close_pos                    = (sign_holding > 0) * -current_holding
                close_time[j  + close_pos:j] = time[i]
                close_price[j + close_pos:j] = price[i][0]
                trade_time[j]                = time[i]
                trade_price[j]               = price[i][0]
                long_short[j]                = -1
                buy_sell        = close_pos - 1
                current_holding = current_holding + buy_sell
                j += 1
            else:
                close_pos                    = abs(current_holding)
                close_time[j - close_pos:j]  = time[i]
                close_price[j - close_pos:j] = price[i][0]
                current_holding = 0

        profit       = (close_price - trade_price) * long_short
        trade_record = {'trade_time' : trade_time,
                        'trade_price': trade_price,
                        'close_time' : close_time,
                        'close_price': close_price,
                        'profit'     : profit}

        return trade_record

    @staticmethod
    def get_indices(index, n_hist, n_forward):
        assert n_hist <= index + 1, 'Error:Invalid number of historical observations.'
        start_hist    = index - n_hist + 1
        end_hist      = index + 1
        start_forward = index
        end_forward   = index + n_forward + 1
        return start_hist, end_hist, start_forward, end_forward

    def run_episode(self, index, n_hist, n_forward, trade_th, stop_loss, cl=0.05):
        start_hist, end_hist, start_forward, end_forward = self.get_indices(index, n_hist, n_forward)
        sp = self.calibrate(start_hist, end_hist, cl)
        if sp is not None:
            time, price, order, gross_exp = self.gen_order(start_forward, end_forward, trade_th, stop_loss)
            trade_record = self.gen_trade_record(time, price, order)
            returns      = trade_record['profit'] / gross_exp
            reward       = returns.mean()
        else:
            reward = None
        return reward
