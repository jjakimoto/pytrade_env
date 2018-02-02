import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from urllib.request import urlopen
import json

from pytrade_env.database.config import URL
from pytrade_env.database.sql_declarative import Price30M, Base
from pytrade_env.database.utils import get_data
from pytrade_env.utils import get_time_now, symbol_dict
import pickle


def store(session, ticker, date, open, high, low,
          close, volume, *args, **kwargs):
    obj = Price30M(ticker=ticker, date=date, open=open, high=high,
                   low=low, close=close, volume=volume)
    session.add(obj)
    session.commit()


def store_df(ticker, df):
    # Establish connection
    engine = create_engine(URL)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    df_val = df.values
    for val in tqdm(df_val):
        data = dict(session=session, ticker=ticker)
        for i, col in enumerate(df.columns):
            data[col] = val[i]
        store(**data)
    session.close()


def update(ticker, end=None, period="30m"):
    if end is None:
        end = "1970-01-01 00:00:00"
    df = get_data(ticker, end, end=None, period=period)
    store_df(ticker, df)


if __name__ == '__main__':
    url = "https://api.kraken.com/0/public/AssetPairs"
    res = urlopen(url)
    res = json.loads(res.read())
    pairs = list(res["result"].keys())
    end = get_time_now(False)
    for pair in tqdm(pairs):
        update(pair, period=30)
