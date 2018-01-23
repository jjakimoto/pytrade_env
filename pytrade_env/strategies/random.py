from __future__ import print_function

import datetime
import numpy as np
from copy import deepcopy

from .core import Strategy
from ..events import SignalEvent


class RandomRatioStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        self.strategy_id = 0

    def set(self, bars, events):
        super(RandomRatioStrategy, self).set(bars, events)
        self.prev_actions = np.zeros(shape=len(self.symbols) + 1)
        self.prev_actions[0] = 1.

    def calculate_signals(self, event, *args, **kwargs):
        if event.type == 'MARKET':
            actions = np.random.uniform(size=len(self.symbols) + 1)
            actions = actions / np.sum(actions)
            self.current_actions = deepcopy(actions)
            trade_amount = actions - self.prev_actions
            dt = datetime.datetime.utcnow()
            for i, symbol in enumerate(self.symbols):
                # bar_date = self.bars.get_latest_bar_datetime(s)
                # Index 0 is cash
                val = trade_amount[i + 1]
                sig_dir = ""
                if val < 0:
                    sig_dir = 'SHORT'
                elif val > 0:
                    sig_dir = 'LONG'
                else:
                    continue
                val = np.abs(val)
                signal = SignalEvent(self.strategy_id,
                                     symbol, dt, sig_dir, val)
                self.events.put(signal)

    def update_strategy(self):
        self.prev_actions = deepcopy(self.current_actions)
