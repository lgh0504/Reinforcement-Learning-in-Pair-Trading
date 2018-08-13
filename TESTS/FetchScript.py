from UTILS.FetchAsync import Fetcher
from UTILS.FetchSync import *
import datetime, time


csv_path   = 'YourCsvFilePath'
token      = 'YourToken'
freq       = '1min'
start_date = '2018-01-01'
end_date   = '2018-01-03'
nObs       = 391

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date   = datetime.datetime.strptime(end_date,   '%Y-%m-%d').date()
fetcher    = Fetcher(token)

pairs = [['AAPL', 'MSFT'],
         ['BAC' , 'BK'  ],
         ['C'   , 'WFC' ],
         ['CNP' , 'NEE' ],
         ['F'   , 'GM'  ],
         ['FB'  , 'TWTR'],
         ['IBM' , 'CSCO'],
         ['JNJ' , 'PG'  ],
         ['KO'  , 'PEP' ],
         ['V'   , 'MA'  ]]
ticker_list = [ticker for pair in pairs for ticker in pair]


''' ----- Examples for FetchSync ----- '''

# Example for single ticker.
ticker = 'AAPL'
start  = time.time()
ts     = get_time_series(ticker, start_date, end_date, freq, token, nObs)
end    = time.time()
print('Sync process time for {ticker}: {time}s.'.format(ticker=ticker, time=end-start))

# Example for multiple tickers.
start = time.time()
for ticker in ticker_list:
    ts = get_time_series(ticker, start_date, end_date, freq, token, nObs)
end   = time.time()
print('Sync process time for {n} tickers: {time}s.'.format(n=len(ticker_list), time=end-start))


''' ----- Examples for FetchAsync -----'''

# Example for single ticker.
ticker = 'AAPL'
start  = time.time()
ts     = fetcher.get_time_series(ticker, start_date, end_date, freq, nObs)
end    = time.time()
print('Async process time: {time}s.'.format(time=end-start))


# Example for multiple tickers.
start = time.time()
for ticker in ticker_list:
    ts = fetcher.get_time_series(ticker, start_date, end_date, freq, nObs)
end = time.time()
print('Async process time for {n} tickers: {time}s.'.format(n=len(ticker_list), time=end-start))
