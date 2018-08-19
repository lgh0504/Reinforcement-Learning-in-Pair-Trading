import datetime
import time
import pandas as pd
from UTIL import aioutil, dateutil, fileutil
from FACTORY import source


token      = 'ef79e455ba9b04c3df719407e34f05e1b051b4d6'
freq       = '1min'
start_date = '2018-01-01'
end_date   = '2018-01-03'
n_obs      = 391
ticker     = 'MSFT'
attr       = ['date', 'close']

pairs = [['AAPL', 'MSFT'], ['BAC' , 'BK'  ], ['C'   , 'WFC' ],
         ['CNP' , 'NEE' ], ['F'   , 'GM'  ], ['FB'  , 'TWTR'],
         ['IBM' , 'CSCO'], ['JNJ' , 'PG'  ], ['KO'  , 'PEP' ],
         ['V'   , 'MA'  ]]
ticker_list = [ticker for pair in pairs for ticker in pair]

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date   = datetime.datetime.strptime(end_date,   '%Y-%m-%d').date()

period   = dateutil.get_dates_weekday(start_date, end_date)
data_cls = source.get_src_cls('Tiingo')
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
