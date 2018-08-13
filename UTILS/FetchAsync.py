import pandas as pd
import datetime, calendar, asyncio, aiohttp


class Fetcher:

    def __init__(self, token):
        self.token = token

    async def fetch_intraday_async(self, ticker, target_date, freq, session):
        target_date = target_date.strftime('%Y-%m-%d')
        url = 'https://api.tiingo.com/iex/{ticker}/prices?startDate={target_date}' \
              '&endDate={target_date}&resampleFreq={freq}&&token={token}' \
              .format(ticker     =ticker,
                      target_date=target_date,
                      freq       =freq,
                      token      =self.token)
        async with session.get(url) as response:
                print('Fetching {ticker} data on {date}...'.format(ticker=ticker, date=target_date))
                prices = await response.json()
                print('Completed fetching {ticker} on {date}.'.format(ticker=ticker, date=target_date))
                return prices

    async def create_tasks(self, ticker, start_date, end_date, freq, loop):
        period = self.get_target_dates(start_date, end_date)
        tasks  = []
        async with aiohttp.ClientSession(loop=loop) as session:
            for date in period:
                tasks.append(asyncio.ensure_future(self.fetch_intraday_async(ticker, date, freq, session)))
            await asyncio.gather(*tasks)
        return tasks

    def get_time_series_async(self, ticker, start_date, end_date, freq, nObs):

        loop    = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop    = asyncio.get_event_loop()
        results = loop.run_until_complete(self.create_tasks(ticker, start_date, end_date, freq, loop))
        loop.close()

        ts = pd.DataFrame()
        for result in results:
            prices = pd.DataFrame(result.result())
            if len(prices) != 0:
                prices = self.format_prices(prices, nObs)
                ts     = pd.concat([ts, prices])
        ts.reset_index(drop=True, inplace=True)
        return ts

    @staticmethod
    def get_target_dates(start_date, end_date):
        dt = end_date - start_date
        weekdays = []
        for i in range(dt.days + 1):
            date = start_date + datetime.timedelta(i)
            if calendar.weekday(date.year, date.month, date.day) < 5:
                weekdays.append(date)
        return weekdays

    @staticmethod
    def format_prices(prices, nObs):
        prices = prices[['date', 'close']]
        prices.rename(columns={'date': 'Time', 'close': 'Price'})
        if len(prices) > nObs:
            prices = prices.iloc[:nObs]
        return prices
