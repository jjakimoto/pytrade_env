import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from urllib.request import urlopen
import json
import pickle
from time import sleep

from pytrade_env.database.config import URL, QUANDL_URL
from pytrade_env.database.sql_declarative import Price30M, Base, StockPriceDay
from pytrade_env.database.utils import get_data, get_stock_tickers
from pytrade_env.utils import get_time_now, symbol_dict


def store(session, ticker, date, open, high, low,
          close, volume, table, *args, **kwargs):
    if table == "price30m":
        obj = Price30M(ticker=ticker, date=date, open=open, high=high,
                       low=low, close=close, volume=volume)
    else:
        obj = StockPriceDay(ticker=ticker, date=date, open=open, high=high,
                            low=low, close=close, volume=volume)
    session.add(obj)
    session.commit()


def store_df(ticker, df, table="price30m"):
    # Establish connection
    if table == "stock_price_daily":
        engine = create_engine(QUANDL_URL)
    else:
        engine = create_engine(URL)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    df_val = df.values
    for val in tqdm(df_val):
        data = dict(session=session, ticker=ticker)
        for i, col in enumerate(df.columns):
            data[col] = val[i]
        store(**data, table=table)
    session.close()


def update(ticker, end=None, period="30m", exchange="kraken"):
    if end is None:
        end = "1970-01-01 00:00:00"
    df = get_data(ticker, start=end, end=None,
                  period=period, exchange=exchange)
    if exchange == "stock":
        table = "stock_price_daily"
    else:
        table = "price30m"
    store_df(ticker, df, table=table)


if __name__ == '__main__':
    # url = "https://api.kraken.com/0/public/AssetPairs"
    # res = urlopen(url)
    # res = json.loads(res.read())
    # pairs = list(res["result"].keys())

    path = "/home/tomoaki/work/pytrade_env/pytrade_env/data/ticker1.pkl"
    file = open(path, "rb")
    pairs = pickle.load(file)
    # pairs = get_stock_tickers()
    end = get_time_now(False)
    for pair in tqdm(pairs):
        update(pair, period=1800, exchange="polo")
        sleep(3)
