import unittest
import numpy as np
from copy import deepcopy

from pytrade_env.strategies import RandomRatioStrategy
from pytrade_env.portfolios import RatioPortfolio
# from .utils import backtest

import os

from pytrade_env.runners import Runner


class TestRunner(Runner):
    weights_list = []

    def _update_strategy(self):
        self.strategy.update_strategy()
        self.weights_list.append(self.portfolio.weights_val)


class Context:
    price_keys = ['open', 'high', 'low']
    volume_keys = ['volume', 'quoteVolume']
    initial_capital = 1.0
    commission_rate = None


def backtest(start, end, strategy, portfolio_cls):
    low_volume_ticker = ['USDT_BCH', 'USDT_ZEC']
    # Load data
    data_dir = "/home/tomoaki/work/Development/cryptocurrency/data"
    filenames = os.listdir(data_dir)
    symbols = []
    for name in filenames:
        if '.csv' in name and name.startswith('USD'):
            flag = True
            for tick in low_volume_ticker:
                if name.startswith(tick):
                    flag = False
            if flag:
                symbol = name.split('.')[0]
                symbols.append(symbol)

    context = Context()
    context.start = start
    context.end = end
    runner = TestRunner(strategy, symbols, context,
                        portfolio_cls=portfolio_cls)
    runner.run(start, end)
    return runner


class TestRandomRatioStrategy(RandomRatioStrategy):
    def __init__(self, *args, **kwargs):
        self.actions_list = []
        super().__init__(*args, **kwargs)

    def update_strategy(self):
        self.actions_list.append(deepcopy(self.current_actions))
        super().update_strategy()


class TestRatioPortfolio(unittest.TestCase):

    def test_ratio_portfolio(self):
        start = '2018-01-01 00:00:00'
        end = '2018-01-15 00:00:00'
        runner = backtest(start, end, strategy=TestRandomRatioStrategy(),
                          portfolio_cls=RatioPortfolio)
        weights_list = runner.weights_list
        # print("**********************portfolio")
        # print(weights_list)
        # print("*************************strategy")
        # print(runner.strategy.actions_list)
        for generated_weights, strategy_weights in zip(
                weights_list, runner.strategy.actions_list):
            np.testing.assert_array_almost_equal(generated_weights,
                                                 strategy_weights)


if __name__ == '__main__':
    unittest.main()
