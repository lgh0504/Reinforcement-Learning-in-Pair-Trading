import datetime
import time
import pandas as pd
from UTIL import aioutil, dateutil, fileutil
from MAIN import DataAPI


token      = 'YOUR_TOKEN'
freq       = '1min'
start_date = '2018-01-01'
end_date   = '2018-01-03'
n_obs      = 391
ticker     = 'AAPL'
attr       = ['date', 'close']

pairs = [['AAPL', 'MSFT'], ['BAC' , 'BK'  ], ['C'   , 'WFC' ],
         ['CNP' , 'NEE' ], ['F'   , 'GM'  ], ['FB'  , 'TWTR'],
         ['IBM' , 'CSCO'], ['JNJ' , 'PG'  ], ['KO'  , 'PEP' ],
         ['V'   , 'MA'  ]]
ticker_list = [ticker for pair in pairs for ticker in pair]

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date   = datetime.datetime.strptime(end_date,   '%Y-%m-%d').date()

period   = dateutil.get_dates_weekday(start_date, end_date)
data_cls = DataAPI.get_src_cls('Tiingo')
paras    = [{'url': data_cls.get_url_intraday(ticker, date, freq, token)} for date in period]


# Example of asyncio data fetching.
start   = time.time()
loop    = aioutil.create_loop()
tasks   = aioutil.create_tasks(loop, paras, data_cls.fetch_data_async)
results = loop.run_until_complete(tasks)
loop.close()
output  = pd.DataFrame()
for result in results:
    data = result.result()
    if len(data) != 0:
        data   = data_cls.format_data(data, attr, n_obs)
        output = pd.concat([output, data])
output.reset_index(drop=True, inplace=True)
end = time.time()
print('Async processing time: {time}s.'.format(time=end-start))


# Example of synchronous data fetching.
start   = time.time()
results = [data_cls.fetch_data(para['url']) for para in paras]
output  = pd.DataFrame()
for data in results:
    if len(data) != 0:
        data   = data_cls.format_data(data, attr, n_obs)
        output = pd.concat([output, data])
output.reset_index(drop=True, inplace=True)
end = time.time()
print('Sync processing time: {time}s.'.format(time=end-start))


import pandas as pd
import numpy as np
from MAIN.Strategy import Cointegration
import matplotlib.pyplot as plt
a = pd.read_csv(r'C:\Users\lk258jt\PycharmProjects\Reinforcement-Learning-in-Pair-Trading\FILE\AAPL.csv')
b = pd.read_csv(r'C:\Users\lk258jt\PycharmProjects\Reinforcement-Learning-in-Pair-Trading\FILE\MSFT.csv')
o = Cointegration(a,b,'date','close')
start = 1100
end = 1200
sp = o.calibrate(start,end,0.1)
trade_th = 1
stop_loss = 2
time, price, order = o.gen_order(start, end, trade_th, stop_loss)
trade_record = o.gen_trade_record(time, price, order)


x, y, time  = o.get_sample(start, end)
spread = o.cal_spread(x, y)
spread_t0 = spread[:-1]
spread_t1 = spread[1:]
# x_t1  = x[1:]
# y_t1  = y[1:]
t_t1  = time[1:]


from timeit import default_timer as timer
start = timer()

signals = pd.DataFrame()
signals['time']       = t_t1
signals['order']      = None
# signals['x_t1']       = x_t1
# signals['y_t1']       = y_t1
signals['spread_t1']  = spread_t1

ind_buy  = np.logical_and(spread_t0 >= -trade_th,  spread_t1 <= -trade_th).reshape(-1,)
ind_sell = np.logical_and(spread_t0 <=  trade_th,  spread_t1 >=  trade_th).reshape(-1,)
ind_stop = np.logical_or(np.logical_and(spread_t0 >= -stop_loss, spread_t1 <= -stop_loss).reshape(-1,),
                         np.logical_and(spread_t0 <=  stop_loss, spread_t1 >=  stop_loss).reshape(-1,))
signals['order'].loc[ind_buy]  = 'Buy'
signals['order'].loc[ind_sell] = 'Sell'
signals['order'].loc[ind_stop] = 'Stop'
signals['order'].iloc[-1]      = 'Stop'
signals = signals.loc[signals['order'].notnull(), :]

current_holding = 0
n_buysell = sum(signals['order'] != 'Stop')
trade_record = pd.DataFrame(columns=['trade_time', 'trade_price', 'close_time', 'close_price', 'long_short', 'profit'],
                            index=range(n_buysell))
j = 0

for i in range(len(signals)):
    sign_holding = int(np.sign(current_holding))
    if signals['order'].iloc[i] == 'Buy':
        close_pos = (sign_holding < 0) * -current_holding
        trade_record['close_time' ].iloc[j-close_pos:j] = signals['time'].iloc[i]
        trade_record['close_price'].iloc[j-close_pos:j] = signals['spread_t1'].iloc[i]
        trade_record['trade_time' ].iloc[j]             = signals['time'].iloc[i]
        trade_record['trade_price'].iloc[j]             = signals['spread_t1'].iloc[i]
        trade_record['long_short' ].iloc[j]             = 1
        buy_sell = close_pos + 1
        current_holding = current_holding + buy_sell
        j += 1
    elif signals['order'].iloc[i] == 'Sell':
        close_pos = (sign_holding > 0) * -current_holding
        trade_record['close_time' ].iloc[j+close_pos:j] = signals['time'].iloc[i]
        trade_record['close_price'].iloc[j+close_pos:j] = signals['spread_t1'].iloc[i]
        trade_record['trade_time' ].iloc[j]             = signals['time'].iloc[i]
        trade_record['trade_price'].iloc[j]             = signals['spread_t1'].iloc[i]
        trade_record['long_short' ].iloc[j]             = -1
        buy_sell = close_pos - 1
        current_holding = current_holding + buy_sell
        j += 1
    else:
        close_pos = abs(current_holding)
        trade_record['close_time' ].iloc[j-close_pos:j] = signals['time'].iloc[i]
        trade_record['close_price'].iloc[j-close_pos:j] = signals['spread_t1'].iloc[i]
        current_holding = 0

    trade_record['profit'] = (trade_record['close_price'] - trade_record['trade_price']) * trade_record['long_short']

end = timer()
print(end-start)



start = timer()


ind_buy  = np.logical_and(spread_t0 >= -trade_th,  spread_t1 <= -trade_th).reshape(-1,)
ind_sell = np.logical_and(spread_t0 <=  trade_th,  spread_t1 >=  trade_th).reshape(-1,)
ind_stop = np.logical_or(np.logical_and(spread_t0 >= -stop_loss, spread_t1 <= -stop_loss).reshape(-1,),
                         np.logical_and(spread_t0 <=  stop_loss, spread_t1 >=  stop_loss).reshape(-1,))
order = np.array([None] * len(t_t1))
order[ind_buy]  = 'Buy'
order[ind_sell] = 'Sell'
order[ind_stop] = 'Stop'
order[-1]       = 'Stop'
time_t1            = t_t1[order != None]
price           = spread_t1[order != None]
order           = order[order != None]

current_holding = 0
n_buy_sell = sum(order != 'Stop')
trade_time = np.array([None] * n_buy_sell)
trade_price = np.array([None] * n_buy_sell)
close_time = np.array([None] * n_buy_sell)
close_price = np.array([None] * n_buy_sell)
long_short = np.array([None] * n_buy_sell)

j = 0

for i in range(len(order)):
    sign_holding = int(np.sign(current_holding))
    if order[i] == 'Buy':
        close_pos = (sign_holding < 0) * -current_holding
        close_time[j-close_pos:j] = time_t1[i]
        close_price[j-close_pos:j] = price[i][0]
        trade_time[j]             = time_t1[i]
        trade_price[j]             = price[i][0]
        long_short[j]             = 1
        buy_sell = close_pos + 1
        current_holding = current_holding + buy_sell
        j += 1
    elif order[i] == 'Sell':
        close_pos = (sign_holding > 0) * -current_holding
        close_time[j+close_pos:j] = time_t1[i]
        close_price[j+close_pos:j] = price[i][0]
        trade_time[j]             = time_t1[i]
        trade_price[j]             = price[i][0]
        long_short[j]             = -1
        buy_sell = close_pos - 1
        current_holding = current_holding + buy_sell
        j += 1
    else:
        close_pos = abs(current_holding)
        close_time[j-close_pos:j] = time_t1[i]
        close_price[j-close_pos:j] = price[i][0]
        current_holding = 0

profit = (close_price - trade_price) * long_short
trade_record2 = {'trade_time': trade_time,
             'trade_price': trade_price,
             'close_time': close_time,
             'close_price': close_price,
             'profit': profit}

end = timer()
print(end-start)



