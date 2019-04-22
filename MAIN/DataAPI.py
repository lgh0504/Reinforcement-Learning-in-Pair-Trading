import sys
import pandas as pd


def get_src_cls(source_name):
    return getattr(sys.modules[__name__], source_name)


class Tiingo(object):

    @staticmethod
    def get_url_intraday(ticker, target_date, freq, token):
        url = 'https://api.tiingo.com/iex/{ticker}/prices?startDate={target_date}' \
              '&endDate={target_date}&resampleFreq={freq}&&token={token}' \
            .format(ticker=ticker,
                    target_date=target_date,
                    freq=freq,
                    token=token)
        return url

    @staticmethod
    def get_url_realtime(ticker):
        url = 'https://api.tiingo.com/iex/{ticker}'.format(ticker=ticker)
        return url

    @staticmethod
    def get_url_daily(ticker, start_date, end_date, token):
        url = 'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}' \
              '&endDate={end_date}&token={token}' \
            .format(ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    token=token)
        return url

    @staticmethod
    async def fetch_data_async(session, url):
        async with session.get(url) as response:
            json = await response.json()
            data = pd.DataFrame(json)
            return data

    @staticmethod
    def fetch_data(url):
        data = pd.read_json(url)
        return data

    @staticmethod
    def format_data(data, attr, n_obs):
        data = data[attr]
        if len(data) > n_obs:
            data = data.iloc[:n_obs]
        return data
