from __future__ import print_function

import datetime
import numpy as np
from copy import deepcopy

from .core import Strategy
from ..events import SignalEvent


class RandomRatioStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        self.strategy_id = 0

    def calculate_signals(self, event, *args, **kwargs):
        if event.type == 'MARKET':
            actions = np.random.uniform(size=len(self.symbols) + 1)
            actions = actions / np.sum(actions)
            self.current_actions = deepcopy(actions)
            dt = datetime.datetime.utcnow()
            for i, symbol in enumerate(self.symbols):
                # bar_date = self.bars.get_latest_bar_datetime(s)
                # Index 0 is cash
                val = actions[i + 1]
                val = np.abs(val)
                sig_dir = ""
                signal = SignalEvent(self.strategy_id,
                                     symbol, dt, sig_dir, val)
                self.events.put(signal)
