from .core import BasePortfolio


class RatioPortfolio(BasePortfolio):
    def get_quantity(self, symbol, value):
        total = self.asset_size
        new_pos = total * value / self.bars.get_latest_market_value(symbol)
        return new_pos - self.current_positions[symbol]
