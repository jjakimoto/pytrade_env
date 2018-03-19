from __future__ import print_function

import datetime
import numpy as np
from copy import deepcopy
from six.moves import xrange

from .core import Strategy
from ..events import SignalEvent


class AgentWrapper(Strategy):
    def __init__(self, agent, num_epochs=1):
        self.agent = agent
        self.strategy_id = 0
        self.prev_bars = None
        self.prev_actions = np.zeros(shape=self.agent.action_shape)
        self.prev_actions[0] = 1.
        self.num_epochs = num_epochs
        self.current_actions = deepcopy(self.agent.get_recent_actions())
        self.prev_actions = deepcopy(self.current_actions)

    def calculate_signals(self, event, *args, **kwargs):
        # Update model before predicting actions
        self._update_agent()
        if event.type == 'MARKET':
            recent_state = self.agent.get_recent_state()
            recent_actions = self.agent.get_recent_actions()
            actions = self.agent.predict(recent_state, recent_actions)
            # Update actions
            self.prev_actions = deepcopy(self.current_actions)
            self.current_actions = deepcopy(actions)
            dt = datetime.datetime.utcnow()
            for i, symbol in enumerate(self.symbols):
                # bar_date = self.bars.get_latest_bar_datetime(s)
                # Index 0 is cash
                val = actions[i + 1]
                # Update signal_dir at portfolio
                sig_dir = ""
                val = np.abs(val)
                signal = SignalEvent(self.strategy_id,
                                     symbol, dt, sig_dir, val)
                self.events.put(signal)

    def _update_agent(self):
        current_bars = self.bars.get_current_bars()
        if self.prev_bars is None:
            self.prev_bars = deepcopy(current_bars)
        returns = current_bars['price'][:, 0] / self.prev_bars['price'][:, 0] - 1.
        # observation = self._get_observation(current_bars)
        observation = deepcopy(current_bars)
        trade_amount = np.sum(np.abs(self.current_actions[1:] - self.prev_actions[1:]))
        reward = np.sum(returns * self.current_actions[1:])
        """
        # We do not calculate actual mu for speeding up
        if not is_training:
            mu = calculate_pv_after_commission(self.current_actions,
                                               self.prev_actions,
                                               self.commission_rate)
            reward = mu * (reward + 1.) - 1.
            cost = 1 - mu
        else:
            cost = 0
        """
        cost = 0
        self.prev_actions = deepcopy(self.current_actions)
        self.prev_bars = deepcopy(current_bars)
        info = {
            'reward': reward,
            'returns': returns,
            'cost': cost,
            'trade_amount': trade_amount,
            'time': self.bars.get_latest_bar_datetime(),
        }
        terminal = False

        response = self.agent.observe(observation, self.current_actions,
                                      reward, terminal, info,
                                      training=False, is_store=True)
        for epoch in xrange(self.num_epochs):
            # Update parameters
            response = self.agent.nonobserve_learning(use_newest=True)
