from __future__ import print_function

import numpy as np
import pandas as pd
from collections import defaultdict

from .core import DataHandler
from ..events import MarketEvent
from ..database.fetch import fetch_data
from ..utils import date2datetime, datetime2date, get_time_now


class HistoricSQLDataHandler(DataHandler):
    """
    DataHanderl to interact with SQL historical database

    Parameters
    ----
    events: Queue, Event queue
    symbol_list: list(str)
    start: str
        yyyy:mm:dd hh-mm-ss format
    end: str, Optional
    keys: list(str),
        column names used by handler. Need to put a key used for market value
        as the first element.
    """

    def __init__(self, events, symbol_list,
                 keys=['open', 'high', 'low', 'volume']):
        self.events = events
        self.symbol_list = symbol_list
        self.keys = keys
        self.market_value_key = self.keys[0]
        self.symbol_data = dict()
        self.latest_symbol_data = defaultdict(lambda: [])
        self.continue_trading = True

    def set_trange(self, start, end=None):
        """You have to initialize function before using"""
        data = fetch_data(start, end, self.symbol_list_list)
        # Build imputed data with columns key
        self.col_data = defaultdict(lambda: [])
        for symbol, val in data.items():
            df = pd.DataFrame(val.values,
                              index=val.index, columns=val.columns)
            df = df.loc[~df.index.duplicated(keep='first')]
            for col in val.columns:
                self.col_data[col].append(df[col])
        for col in self.col_data.keys():
            df = pd.concat(self.col_data[col], axis=1, keys=self.symbol_list)
            df.interpolate(method='linear',
                           limit_direction='both',
                           inplace=True)
            self.col_data[col] = df
        self.allow_time_index = df.index

        # Redefine time range within allowed time index
        start = date2datetime(start)
        self.start = str(max(start, self.allow_time_index[0]))
        if end is None:
            self.end = self.allow_time_index[-1]
        else:
            end = date2datetime(end)
            self.end = str(min(end, self.allow_time_index[-1]))

        print('start:', self.start)
        print('end:', self.end)

        # Store imputed data with symbol keys
        data_array = []
        for symbol in self.symbol_list:
            val = []
            for col in self.keys:
                df = self.col_data[col][[symbol]]
                val.append(df.values)
            self.time_index = df.index
            if val:
                val = np.concatenate(val, axis=1)
            data_array.append(np.expand_dims(val, 1))
            self.symbol_data[symbol] = pd.DataFrame(val,
                                                    columns=self.keys,
                                                    index=self.time_index)

        self.data_array = np.concatenate(data_array, axis=1)
        # Idx for fetching new bar
        self.idxes = dict((symbol, 0) for symbol in self.symbol_list)
        self.max_idx = len(self.time_index) - 1

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        idx = self.idxes[symbol]
        if idx <= self.max_idx:
            data = self.symbol_data[symbol].iloc[idx]
            time = self.time_index[idx]
            # Update index
            self.idxes[symbol] += 1
            return dict(time=time, data=data)
        else:
            raise StopIteration()

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]["data"]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]["data"]

    def get_latest_bar_datetime(self, symbol=None):
        """
        Returns a Python datetime object for the last bar.
        """
        if symbol is None:
            symbol = self.symbol_list[0]
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]['time']

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            if val_type in self.keys:
                return getattr(bars_list[-1]["data"], val_type)
            else:
                raise NotImplementedError("No implementation for val_type={}".format(val_type))

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            if val_type in self.keys:
                return np.array([getattr(b["data"], val_type) for b in bars_list])
            else:
                raise NotImplementedError("No implementation for val_type={}".format(val_type))

    def update_bars(self, is_initial=False):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = self._get_new_bar(s)
            except StopIteration:
                self.continue_trading = False
                bar = None
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar["data"])
            self.events.put(MarketEvent())

    def get_latest_market_value(self, symbol):
        return self.get_latest_bar_value(symbol, self.market_value_key)

    def get_latest_market_values(self, symbol, N=1):
        return self.get_latest_bars_values(symbol, self.market_value_key, N=N)
