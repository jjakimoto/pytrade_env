import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm

from pytrade_env.utils import date2seconds, seconds2datetime, get_time_now


Time2Seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
                "3h": 3600 * 3, "6h": 3600 * 6, "12h": 3600 * 12,
                "1D": 3600 * 24, "7D": 3600 * 24 * 7}


def get_data(ticker, start, end=None, period="30m", exchange="bitfx"):
    # We do not want to include start time
    start_sc = date2seconds(start) + 1
    if end is None:
        end = get_time_now()
    end_sc = date2seconds(end)
    if exchange == "polo":
        base_url = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=%d"

        url = base_url % (ticker, start_sc, end_sc, period)
        df = pd.read_json(url)
    elif exchange == "bitfx":
        base_url = "https://api.bitfinex.com/v2/candles/trade:%s:%s/hist?start=%d&end=%d"
        limit = 120
        period_sc = Time2Seconds[period]
        step = period_sc * limit
        ends_sc = np.arange(end_sc, start_sc, -step)
        dfs = []
        print(start_sc, end_sc)
        print(start, end)
        for _end_sc in tqdm(ends_sc):
            _start_sc = max(start_sc, _end_sc - step)
            url = base_url % (period, ticker, int(_start_sc * 1000), int(_end_sc * 1000))
            while True:
                try:
                    df = pd.read_json(url)
                    break
                except Exception as e:
                    print(e)
                    if int(e.status) == 429:
                        print('hit rate limit, sleeping for a minute...')
                        sleep(60)
            if len(df) == 0:
                break
            else:
                df = _preprocess_bitfx(df)
                dfs.append(df)
                # You can hit ~ 20 time each minute
                sleep(3)
        if dfs:
            df = pd.concat(dfs)
        else:
            df = pd.DataFrame(dfs)
    else:
        raise NotImplementedError()
    return df


def _preprocess_bitfx(df):
    date = [seconds2datetime(x / 1000) for x in df[0].values]
    df_dict = dict(date=date,
                   open=df[1].values,
                   close=df[2].values,
                   high=df[3].values,
                   low=df[4].values,
                   volume=df[5].values)
    df = pd.DataFrame(df_dict)
    return df
