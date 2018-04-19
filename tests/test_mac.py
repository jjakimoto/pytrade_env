import unittest
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from pytrade_env.strategies import MovingAverageCrossStrategy
from pytrade_env.portfolios import Portfolio
from utils import backtest


class TestMACStrategy(MovingAverageCrossStrategy):
    def __init__(self, *args, **kwargs):
        self.actions_list = []
        super().__init__(*args, **kwargs)

    def calculate_signals(self, event, *args, **kwargs):
        super().calculate_signals(event, *args, **kwargs)
        self.actions_list.append(deepcopy(self.current_actions))


class TestPortfolio(unittest.TestCase):
    def test_ratio_portfolio(self):
        start = '2018-01-14 00:00:00'
        end = '2018-01-15 00:00:00'
        strategy = TestMACStrategy()
        runner = backtest(start, end, strategy=strategy,
                          portfolio_cls=Portfolio)
        # Check if rebalance is executed correctly
        positions_list = runner.positions_list
        new_positions_list = []
        for i, actions in enumerate(strategy.actions_list):
            positions = dict()
            old_positions = positions_list[i]
            for key in old_positions.keys():
                if key in actions:
                    positions[key] = actions[key] + old_positions[key]
                else:
                    positions[key] = old_positions[key]
            new_positions_list.append(positions)
        for gp, sp in tqdm(zip(positions_list, new_positions_list)):
            for key in gp.keys():
                self.assertAlmostEqual(gp[key], sp[key])
        # Check if portfolio value is udpated
        result = runner.equity_curve["total"].values
        abs_val = np.max(np.abs(result - np.mean(result)))
        self.assertNotAlmostEqual(abs_val, 0)


if __name__ == '__main__':
    unittest.main()
