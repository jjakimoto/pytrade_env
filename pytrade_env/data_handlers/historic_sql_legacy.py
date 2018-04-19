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
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, events, symbols,
                 price_keys=['open', 'high', 'low'],
                 volume_keys=['volume']):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        ’symbol.csv’, where symbol is a string in the list.
        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbols - A list of symbol strings.
        """

        self.events = events
        self.symbols = symbols
        self.price_keys = price_keys
        self.market_value_key = self.price_keys[0]
        self.volume_keys = volume_keys
        self.latest_symbol_data = defaultdict(lambda: [])
        self.continue_trading = True

    def set_trange(self, start, end=None):
        data = fetch_data(start, end, self.symbols)
        # Build imputed data with columns key
        self.col_data = defaultdict(lambda: [])
        for symbol, val in data.items():
            df = pd.DataFrame(val.values,
                              index=val.index, columns=val.columns)
            df = df.loc[~df.index.duplicated(keep='first')]
            for col in val.columns:
                self.col_data[col].append(df[col])
        for col in self.col_data.keys():
            df = pd.concat(self.col_data[col], axis=1, keys=self.symbols)
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
        price_data = {}
        price_data_val = []
        for symbol in self.symbols:
            val = []
            for col in self.price_keys:
                df = self.col_data[col][[symbol]]
                val.append(df.values)
            self.time_index = df.index
            if val:
                val = np.concatenate(val, axis=1)
            price_data_val.append(np.expand_dims(val, 1))
            price_data[symbol] = pd.DataFrame(val, columns=self.price_keys,
                                              index=self.time_index)

        # Store imputed data with symbol keys
        volume_data = {}
        volume_data_val = []
        for symbol in self.symbols:
            val = []
            for col in self.volume_keys:
                df = self.col_data[col][[symbol]]
                val.append(df.values)
            self.time_index = df.index
            if val:
                val = np.concatenate(val, axis=1)
            volume_data_val.append(np.expand_dims(val, 1))
            volume_data[symbol] = pd.DataFrame(val, columns=self.volume_keys,
                                               index=self.time_index)

        self.price_data = price_data
        self.price_data_val = np.concatenate(price_data_val, axis=1)
        self.volume_data = volume_data
        self.volume_data_val = np.concatenate(volume_data_val, axis=1)
        # Idx for fetching new bar
        self.idxes = dict((symbol, 0) for symbol in self.symbols)
        self.max_idx = len(self.time_index) - 1

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        idx = self.idxes[symbol]
        if idx <= self.max_idx:
            price = self.price_data[symbol].iloc[idx]
            volume = self.volume_data[symbol].iloc[idx]
            time = self.time_index[idx]
            # Update index
            self.idxes[symbol] += 1
            return dict(time=time, price=price, volume=volume)
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
            return bars_list[-1]

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
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol=None):
        """
        Returns a Python datetime object for the last bar.
        """
        if symbol is None:
            symbol = self.symbols[0]
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
            if val_type in self.price_keys:
                return getattr(bars_list[-1]['price'], val_type)
            elif val_type in self.volume_keys:
                return getattr(bars_list[-1]['volume'], val_type)
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
            if val_type in self.price_keys:
                return np.array([getattr(b['price'], val_type) for b in bars_list])
            elif val_type in self.volume_keys:
                return np.array([getattr(b['volume'], val_type) for b in bars_list])
            else:
                raise NotImplementedError("No implementation for val_type={}".format(val_type))

    def update_bars(self, is_initial=False):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbols:
            try:
                bar = self._get_new_bar(s)
            except StopIteration:
                self.continue_trading = False
                bar = None
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        if not is_initial:
            self.events.put(MarketEvent())

    def update_data(self):
        new_end = None
        # _end = datetime2date(self.end)
        data = fetch_data(self.end, new_end, self.symbols)
        if not data:
            return None
        # Build imputed data with columns key
        _col_data = defaultdict(lambda: [])
        for symbol, val in data.items():
            df = pd.DataFrame(val.values,
                              index=val.index, columns=val.columns)
            df = df.loc[~df.index.duplicated(keep='first')]
            for col in val.columns:
                _col_data[col].append(df[col])
        for col in _col_data.keys():
            df = pd.concat(_col_data[col], axis=1, keys=self.symbols)
            df.interpolate(method='linear',
                           limit_direction='both',
                           inplace=True)
            self.col_data[col] = self.col_data[col].append(df)
        self.df = df
        self.allow_time_index = self.allow_time_index.append(df.index)

        # Redefine time range within allowed time index
        self.end = self.allow_time_index[-1]

        print('start:', self.start)
        print('end:', self.end)

        # Store imputed data with symbol keys
        price_data = {}
        price_data_val = []
        for symbol in self.symbols:
            val = []
            for col in self.price_keys:
                df = self.col_data[col][[symbol]]
                val.append(df.values)
            self.time_index = df.index
            if val:
                val = np.concatenate(val, axis=1)
            price_data_val.append(np.expand_dims(val, 1))
            price_data[symbol] = pd.DataFrame(val, columns=self.price_keys,
                                              index=self.time_index)

        # Store imputed data with symbol keys
        volume_data = {}
        volume_data_val = []
        for symbol in self.symbols:
            val = []
            for col in self.volume_keys:
                df = self.col_data[col][[symbol]]
                val.append(df.values)
            self.time_index = df.index
            if val:
                val = np.concatenate(val, axis=1)
            volume_data_val.append(np.expand_dims(val, 1))
            volume_data[symbol] = pd.DataFrame(val, columns=self.volume_keys,
                                               index=self.time_index)

        self.price_data = price_data
        self.price_data_val = np.concatenate(price_data_val, axis=1)
        self.volume_data = volume_data
        self.volume_data_val = np.concatenate(volume_data_val, axis=1)
        self.max_idx = len(self.time_index) - 1
        self.continue_trading = True

    def get_latest_market_value(self, symbol):
        return self.get_latest_bar_value(symbol, self.market_value_key)

    def get_latest_market_values(self, symbol, N=1):
        return self.get_latest_bars_values(symbol, self.market_value_key, N=N)

    def _get_current_price_array(self):
        current_prices = []
        for symbol in self.symbols:
            price = self.get_latest_bar(symbol)['price'].values
            current_prices.append(price)
        return np.array(current_prices)

    def _get_current_volume_array(self):
        current_volumes = []
        for symbol in self.symbols:
            volume = self.get_latest_bar(symbol)['volume'].values
            current_volumes.append(volume)
        return np.array(current_volumes)

    def get_current_bars(self):
        price = self._get_current_price_array()
        volume = self._get_current_volume_array()
        return dict(price=price, volume=volume)

    def _get_prev_price_array(self):
        prev_prices = []
        for symbol in self.symbols:
            price = self.get_latest_bars(symbol, 2)[0]['price'].values
            prev_prices.append(price)
        return np.array(prev_prices)

    def _get_prev_volume_array(self):
        prev_volumes = []
        for symbol in self.symbols:
            volume = self.get_latest_bars(symbol, 2)[0]['volume'].values
            prev_volumes.append(volume)
        return np.array(prev_volumes)

    def get_prev_bars(self):
        price = self._get_prev_price_array()
        volume = self._get_prev_volume_array()
        return dict(price=price, volume=volume)
