from .core import BaseRunner
from ..data_handlers import HistoricSQLDataHandler
from ..executions import SimulatedExecutionHandler
from ..portfolios import Portfolio


class Runner(BaseRunner):
    def __init__(self, strategy, symbols, context,
                 data_handler_cls=HistoricSQLDataHandler,
                 execution_handler_cls=SimulatedExecutionHandler,
                 portfolio_cls=Portfolio):
        super().__init__(strategy, symbols, context, data_handler_cls,
                         execution_handler_cls, portfolio_cls)

    def _calc_market(self, event):
        self.strategy.calculate_signals(event)
        self.portfolio.update_timeindex(event)

    def _calc_signal(self, event):
        self.portfolio.update_signal(event)

    def _calc_order(self, event):
        self.execution_handler.execute_order(event)

    def _calc_fill(self, event):
        self.portfolio.update_fill(event)
        self.strategy.update_fill(event)

    def _update_strategy(self):
        self.strategy.update_strategy()
