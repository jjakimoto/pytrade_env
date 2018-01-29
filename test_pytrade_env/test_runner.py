import unittest
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from pytrade_env.strategies import RandomRatioStrategy
from pytrade_env.portfolios import RatioPortfolio
from test_pytrade_env.utils import backtest


class TestRandomRatioStrategy(RandomRatioStrategy):
    def __init__(self, *args, **kwargs):
        self.actions_list = []
        super().__init__(*args, **kwargs)

    def calculate_signals(self, event, *args, **kwargs):
        super().calculate_signals(event, *args, **kwargs)
        self.actions_list.append(deepcopy(self.current_actions))


class TestRatioPortfolio(unittest.TestCase):

    def test_ratio_portfolio(self):
        start = '2018-01-14 00:00:00'
        end = '2018-01-15 00:00:00'
        runner = backtest(start, end, strategy=TestRandomRatioStrategy(),
                          portfolio_cls=RatioPortfolio)
        # Check if rebalance is actually executed
        weights_list = runner.weights_list
        for generated_weights, strategy_weights in tqdm(zip(
                weights_list, runner.strategy.actions_list)):
            np.testing.assert_array_almost_equal(generated_weights,
                                                 strategy_weights)
        # Check if portfolio value is udpated
        result = runner.equity_curve["total"].values
        abs_val = np.max(np.abs(result - np.mean(result)))
        self.assertNotAlmostEqual(abs_val, 0)


if __name__ == '__main__':
    unittest.main()
