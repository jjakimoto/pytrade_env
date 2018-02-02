import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

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
    filepath = "/home/tomoaki/work/pytrade_env/pytrade_env/data/ticker1.pkl"
    file = open(filepath, "rb")
    tickers = pickle.load(file)
    pairs = [symbol_dict[pair] for pair in tickers]
    end = get_time_now(False)
    for pair in tqdm(pairs):
        update(pair)
