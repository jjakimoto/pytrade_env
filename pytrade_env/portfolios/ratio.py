from .core import BasePortfolio


class RatioPortfolio(BasePortfolio):
    def get_quantity(self, symbol, value):
        total = self.current_holdings['total']
        return total * value / self.bars.get_latest_market_value(symbol)
