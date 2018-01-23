from __future__ import print_function

from abc import ABCMeta, abstractmethod
try:
    import Queue as queue
except ImportError:
    import queue
import numpy as np
from copy import deepcopy


class Env(object):
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(self, symbols, context, data_handler_cls,
                 execution_handler_cls, portfolio_cls):
        """
        Initialises the backtest.

        Args:
            csv_dir: str, The hard root to the CSV data directory.
            symbol_list: list(str), The list of symbol strings.
            intial_capital: float, The starting capital for the portfolio.
            heartbeat: float, Backtest "heartbeat" in seconds
            start_date: datetime.datetime, The start datetime of the strategy.
            data_handler: class, Handles the market data feed.
            execution_handler: class, Handles the orders/fills for trades.
            portfolio: class, Keeps track of portfolio current
                and prior positions.
            strategy: class, Generates signals based on market data.
        """
        self.symbols = symbols
        self.initial_capital = context.initial_capital
        self.commission_rate = context.commission_rate
        self.context = context
        self.price_keys = context.price_keys
        self.volume_keys = context.volume_keys
        self.data_handler_cls = data_handler_cls
        self.execution_handler_cls = execution_handler_cls
        self.portfolio_cls = portfolio_cls

    def set_trange(self, start, end):
        self._start = start
        self._end = end

    def reset(self):
        self._generate_instances()

    def _generate_instances(self):
        # The count number of each event instances
        self.events = queue.Queue()
        self.data_handler = self.data_handler_cls(self.events, self.symbols,
                                                  self.context.price_keys,
                                                  self.context.volume_keys)
        self.data_handler.set_trange(self._start, self._end)
        self.start = self.data_handler.start
        self.end = self.data_handler.end
        # Initialize bar
        self.data_handler.update_bars()
        self.portfolio = self.portfolio_cls(self.data_handler, self.events,
                                            self.start, self.initial_capital)
        self.execution_handler = self.execution_handler_cls(self.events)

    @property
    def time_index(self):
        return self.data_handler.time_index

    @property
    def num_stocks(self):
        return len(self.symbols)

    @property
    def feature_dim(self):
        return len(self.price_keys) + len(self.volume_keys)


class RLEnv(Env, metaclass=ABCMeta):
    def reset(self):
        super().reset()
        self.current_time = self.start
        self.current_step = 0
        observation = self.get_current_bars()
        self.prev_bars = deepcopy(observation)
        self.prev_action = np.array([1.] + list(np.zeros(self.action_dim - 1)))
        return observation

    @abstractmethod
    def step(self, action, is_training=True, *args, **kwargs):
        raise NotImplementedError()

    @property
    def action_dim(self):
        return len(self.symbols) + 1
