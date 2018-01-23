import numpy as np
from copy import deepcopy

from .core import RLEnv
from ..utils import calculate_pv_after_commission
from ..data_handlers import HistoricSQLDataHandler
from ..executions import SimulatedExecutionHandler
from ..portfolios import RatioPortfolio


class LiveTradeRLEnv(RLEnv):
    def __init__(self, symbols, context,
                 data_handler_cls=HistoricSQLDataHandler,
                 execution_handler_cls=SimulatedExecutionHandler,
                 portfolio_cls=RatioPortfolio):
        super().__init__(symbols, context, data_handler_cls,
                         execution_handler_cls, portfolio_cls)

    def step(self, action, is_training=True, *args, **kwargs):
        current_bars = self._get_current_bars()
        returns = current_bars['price'][:, 0] / self.prev_bars['price'][:, 0] - 1.
        # observation = self._get_observation(current_bars)
        observation = deepcopy(current_bars)
        terminal = self._get_terminal()
        trade_amount = np.sum(np.abs(action[1:] - self.prev_action[1:]))
        reward = np.sum(returns * action[1:])
        # We do not calculate actual mu for speeding up
        if not is_training:
            mu = calculate_pv_after_commission(action,
                                               self.prev_action,
                                               self.commission_rate)
            reward = mu * (reward + 1.) - 1.
            cost = 1 - mu
        else:
            cost = 0
        self.prev_action = deepcopy(action)
        self.prev_bars = deepcopy(current_bars)
        info = {
            'reward': reward,
            'returns': returns,
            'cost': cost,
            'trade_amount': trade_amount,
            'time': self._get_time(),
        }
        if not terminal:
            # Update bars
            self._update_time()
        return observation, reward, terminal, info

    def _get_current_price_array(self):
        current_prices = []
        for symbol in self.symbols:
            price = self.data_handler.get_latest_bar(symbol)['price'].values
            current_prices.append(price)
        return np.array(current_prices)

    def _get_current_volume_array(self):
        current_volumes = []
        for symbol in self.symbols:
            volume = self.data_handler.get_latest_bar(symbol)['volume'].values
            current_volumes.append(volume)
        return np.array(current_volumes)

    def _get_current_bars(self):
        price = self._get_current_price_array()
        volume = self._get_current_volume_array()
        return dict(price=price, volume=volume)

    def _update_time(self):
        self.current_step += 1
        self.current_time = self.time_index[self.current_step]
        self.data_handler.update_bars()

    def _get_terminal(self):
        return self.current_time >= self.end

    def _get_time(self):
        return self.current_time
