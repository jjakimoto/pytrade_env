import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm
from urllib.request import urlopen
import json
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile

from pytrade_env.utils import date2seconds, seconds2datetime, get_time_now, symbol_kraken2polo, date2daily
from pytrade_env.constants import QUANDL_APIKEY


stock_columns_map = {
    'Date': "date",
    'Adj. Open': "open",
    'Adj. High': "high",
    'Adj. Low': "low",
    'Adj. Close': "close",
    'Adj. Volume': "volume"
 }


Time2Seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
                "3h": 3600 * 3, "6h": 3600 * 6, "12h": 3600 * 12,
                "1D": 3600 * 24, "7D": 3600 * 24 * 7}


def get_data(ticker, start, end=None, period=30, exchange="kraken"):
    # We do not want to include start time
    start_sc = date2seconds(start) + 1
    if end is None:
        end = get_time_now()
    end_sc = date2seconds(end)
    print(start_sc, end_sc)
    print(start_sc, end_sc)
    print(start, end)
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
    elif exchange == "kraken":
        base_url = "https://api.kraken.com/0/public/OHLC?pair=%s&interval=%d&since=%s"
        url = base_url % (ticker, period, start_sc)
        while True:
            try:
                res = urlopen(url)
                res = json.loads(res.read())
                data = res["result"][ticker]
                break
            except Exception as e:
                print(e)
                print(res)
                if res["error"][0] == 'EService:Unavailable':
                    print('Failed! Try again')
                    sleep(6)
        df = _preprocess_kraken(data)
        _start_sc = data[0][0]
        period_sc = period * 60
        if start_sc <= _start_sc - period_sc and ticker in symbol_kraken2polo:
            _start_sc -= 1
            base_url = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=%d"
            _ticker = symbol_kraken2polo[ticker]
            url = base_url % (_ticker, start_sc, _start_sc, period_sc)
            polo_df = pd.read_json(url)
            # Price
            ohlc_df = df[["open", "high", "low", "close"]]
            polo_ohlc_df = polo_df[["open", "high", "low", "close"]]
            price_ratio = ohlc_df["open"].values[0] / polo_ohlc_df["close"].values[-1]
            polo_ohlc_df *= price_ratio
            ohlc_df = pd.concat([polo_ohlc_df, ohlc_df])
            # Volume
            volume_df = df[["volume"]]
            polo_volume_df = polo_df[["volume"]]
            volume_ratio = volume_df.values[0][0] / polo_volume_df.values[-1][0]
            polo_volume_df *= volume_ratio
            volume_df = pd.concat([polo_volume_df, volume_df])
            # Date
            date_df = pd.concat([polo_df[["date"]], df[["date"]]])
            df = pd.concat([date_df, ohlc_df, volume_df], axis=1)
            df = df.reset_index()
    elif exchange == "stock":
        url = "https://www.quandl.com/api/v3/datasets/%s/data.json?api_key=%s" % (ticker, QUANDL_APIKEY)
        if start is None:
            start = date2daily(start)
            url += "start_date=%s" % start
        if end is None:
            end = date2daily(end)
            url += "start_date=%s" % end
        res = urlopen(url)
        res = json.loads(res.read())
        cols = res['dataset_data']['column_names']
        data = res['dataset_data']['data']
        data_dict = defaultdict(list)
        for x in data:
            for i, col in enumerate(cols):
                if col in stock_columns_map:
                    col = stock_columns_map[col]
                data_dict[col].append(x[i])
        df = pd.DataFrame(data_dict)
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


def _preprocess_kraken(data):
    columns = ["date", "open", "high", "low", "close",
               "vwap", "volume", "count"]
    new_data = defaultdict(list)
    data = deepcopy(data)
    for x in data:
        for i, col in enumerate(columns):
            if i == 0:
                x[i] = seconds2datetime(x[i])
            else:
                x[i] = float(x[i])
            new_data[col].append(x[i])
    df = pd.DataFrame(new_data)
    return df


def get_stock_tickers():
    url = "https://www.quandl.com/api/v3/databases/WIKI/codes.json?api_key=%s" % QUANDL_APIKEY
    res = urlopen(url)
    zipfile = ZipFile(BytesIO(res.read()))
    zipfile.extract(zipfile.namelist()[0])
    df = pd.read_csv(zipfile.namelist()[0], header=None)
    return df[0].values
