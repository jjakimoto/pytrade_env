import os

from pytrade_env.runners import Runner


class TestRunner(Runner):
    weights_list = []

    def _update_strategy(self):
        self.strategy.update_strategy()
        self.weights.append(self.portfolio.weights)


class Context:
    price_keys = ['open', 'high', 'low']
    volume_keys = ['volume', 'quoteVolume']
    initial_capital = 1.0


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
