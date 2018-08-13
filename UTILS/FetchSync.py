import pandas as pd
import datetime, calendar


def get_target_dates(start_date, end_date):
    dt       = end_date - start_date
    weekdays = []
    for i in range(dt.days + 1):
        date = start_date + datetime.timedelta(i)
        if calendar.weekday(date.year, date.month, date.day) < 5:
            weekdays.append(date)
    return weekdays


def format_prices(prices, nObs):
    prices = prices[['date','close']]
    prices.rename(columns={'date':'Time', 'close':'Price'})
    if len(prices) > nObs:
        prices = prices.iloc[:nObs]
    return prices


def fetch_intraday(ticker, target_date, freq, token):
    target_date = target_date.strftime('%Y-%m-%d')
    url    = 'https://api.tiingo.com/iex/{ticker}/prices?startDate={target_date}' \
             '&endDate={target_date}&resampleFreq={freq}&&token={token}'\
             .format(ticker=ticker,
                     target_date=target_date,
                     freq=freq,
                     token=token)
    print('Fetching {ticker} data on {date}...'.format(ticker=ticker, date=target_date))
    prices = pd.read_json(url)
    print('Completed fetching {ticker} on {date}.'.format(ticker=ticker, date=target_date))
    return prices


def get_time_series(ticker, start_date, end_date, freq, token, nObs):
    period  = get_target_dates(start_date, end_date)
    results = []
    ts      = pd.DataFrame()
    for date in period:
        prices = fetch_intraday(ticker, date, freq, token)
        results.append(prices)
    for prices in results:
        if len(prices) != 0:
            prices = format_prices(prices, nObs)
            ts = pd.concat([ts, prices])
    ts.reset_index(drop=True, inplace=True)
    return ts
