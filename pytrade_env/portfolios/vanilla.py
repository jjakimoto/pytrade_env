from .core import BasePortfolio


class Portfolio(BasePortfolio):
    def get_quantity(self, symbol, value):
        return value
