import numpy as np
from copy import deepcopy
import datetime

from .runner import Runner
from ..data_handlers import HistoricSQLDataHandler
from ..executions import SimulatedExecutionHandler
from ..portfolios import RatioPortfolio
from ..events import SignalEvent
from ..strategies import PlaneStrategy


class RLEnv(Runner):
    def __init__(self, symbols, context,
                 data_handler_cls=HistoricSQLDataHandler,
                 execution_handler_cls=SimulatedExecutionHandler,
                 portfolio_cls=RatioPortfolio):
        strategy = PlaneStrategy()
        self.strategy_id = 0
        super().__init__(strategy, symbols, context, data_handler_cls,
                         execution_handler_cls, portfolio_cls)

    def reset(self):
        super().reset()
        self.current_time = self.start
        self.current_step = 0
        # self.data_handler.update_bars(is_initial=True)
        # observation = self.data_handler.get_current_bars()
        observation = None
        self.prev_bars = None
        self.prev_actions = np.array([1.] + list(np.zeros(self.action_dim - 1)))
        return observation

    def step(self, action, is_training=True, *args, **kwargs):
        self.current_actions = action
        self.execute()
        self.data_handler.update_bars()
        current_bars = self.data_handler.get_current_bars()
        if self.prev_bars is None:
            self.prev_bars = deepcopy(current_bars)
        returns = current_bars['price'][:, 0] / self.prev_bars['price'][:, 0] - 1.
        observation = deepcopy(current_bars)
        trade_amount = np.sum(np.abs(self.current_actions[1:] - self.prev_actions[1:]))
        reward = np.sum(returns * self.current_actions[1:])
        cost = 0
        self.prev_actions = deepcopy(self.current_actions)
        self.prev_bars = deepcopy(current_bars)
        info = {
            'reward': reward,
            'returns': returns,
            'cost': cost,
            'trade_amount': trade_amount,
            'time': self.data_handler.get_latest_bar_datetime(),
        }
        terminal = not self.data_handler.continue_trading
        return observation, reward, terminal, info

    def _calc_market(self, event):
        self.portfolio.update_timeindex(event)
        self.calculate_signals(event)

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            dt = datetime.datetime.utcnow()
            for i, symbol in enumerate(self.symbols):
                # Index 0 is cash
                val = self.current_actions[i + 1]
                # Update signal_dir at portfolio
                sig_dir = ""
                val = np.abs(val)
                signal = SignalEvent(self.strategy_id,
                                     symbol, dt, sig_dir, val)
                self.events.put(signal)

    @property
    def time_index(self):
        return self.data_handler.time_index

    @property
    def num_stocks(self):
        return len(self.symbols)

    @property
    def feature_dim(self):
        return len(self.price_keys) + len(self.volume_keys)

    @property
    def action_dim(self):
        return len(self.symbols) + 1
